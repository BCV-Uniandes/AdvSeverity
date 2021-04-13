import time
import numpy as np
import os.path
import torch
import torch.nn as nn

from conditional import conditional
from better_mistakes.model.performance import accuracy

topK_to_consider = (1, 5, 10, 20, 100)

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]


def run(rank, loader, model, loss_function, distances,
        classes, opts, epoch, prev_steps,
        optimizer=None, is_inference=True,
        attack_iters=0, attack_step=1.0, attack_eps=8 / 255,
        attack='none', delta=None, h_utils=None):  # h-free
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """

    topK_to_consider = (1, 5, 10, 20, 100)

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"

    # ===============================================
    # Initialise accumulators to store the several measures of performance (accumulate as sum)
    num_logged = 0
    loss_accum = 0.0
    time_accum = 0.0
    norm_mistakes_accum = 0.0
    flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float)
    hdist_accums = np.zeros(len(topK_to_consider))
    hdist_top_accums = np.zeros(len(topK_to_consider))
    hdist_mistakes_accums = np.zeros(len(topK_to_consider))
    hprecision_accums = np.zeros(len(topK_to_consider))
    hmAP_accums = np.zeros(len(topK_to_consider))

    if is_inference:
        model.eval()
    else:
        model.train()

    # ===============================================
    # initialize numpy confusion matrix to store topks
    topK_to_consider = [i for i in topK_to_consider if i <= h_utils.current_n_classes]
    n = 0
    ns = np.zeros(h_utils.current_n_classes)
    results = {i: np.zeros(h_utils.current_n_classes) for i in topK_to_consider}

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        for batch_idx, (embeddings, target) in enumerate(loader):
            this_load_time = time.time() - time_load0
            this_rest0 = time.time()
            
            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)
            target = target.cuda(opts.gpu, non_blocking=True)

            # ===============================================
            # Natural training step
            # ===============================================

            if attack == 'none': 

                output = model(embeddings)
                loss = loss_function(output, target)

                if not is_inference:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # ===============================================
            # Free adv training step
            # ===============================================

            elif attack == 'free': 

                for rep in range(attack_iters):

                    with torch.no_grad():
                        imgs = (embeddings + delta).detach()
                        imgs.clamp_(0.0, 1.0)
                    imgs.requires_grad = True

                    # get model's prediction
                    logits = model(imgs)

                    if rep == 0:
                        output = logits.detach()

                    loss = loss_function(logits, target)

                    if not is_inference:
                        optimizer.zero_grad()
                        loss.backward()

                        with torch.no_grad():
                            grad = imgs.grad.data
                            delta += attack_step * torch.sign(grad)
                            delta = torch.clamp(delta, -attack_eps, attack_eps)

                        optimizer.step()

            # ===============================================
            # EVALUATION
            # ===============================================

            # start/reset timers
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()

            # only update total number of batch visited for training
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            # compute evaluation
            with torch.no_grad():

                n += target.size(0)
                loss_accum += loss.item() * target.size(0)
                _, topK_pred = output.topk(max(topK_to_consider), 1, True, True)
                correct_pred = topK_pred == target.view(-1, 1)

                for i in target:
                    ns[i.item()] += 1

                for tk in topK_to_consider:
                    topk, _ = correct_pred[:, :tk].max(dim=1)

                    for j, i in enumerate(target):
                        results[tk][i.item()] += int(topk[j].item())


    summary = _generate_summary(n, loss_accum, ns, results, topK_to_consider)

    return summary, tot_steps, delta


def _generate_summary(n, loss_accum, ns, topks, topK_to_consider):

    summary = {}
    summary['loss'] = loss_accum / n

    for topk in topK_to_consider:
        summary[f'accuracy/top{topk}'] = np.mean(topks[topk] / ns).item()

    return summary
