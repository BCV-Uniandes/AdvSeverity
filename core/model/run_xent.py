import time
import numpy as np
import os.path
import torch
import torch.nn as nn
from conditional import conditional
from core.model.performance import accuracy
from core.model.labels import make_batch_onehot_labels, make_batch_soft_labels

topK_to_consider = (1, 5, 10, 20, 100)

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]


def run(loader, model, loss_function, distances,
        classes, opts, epoch, prev_steps, optimizer=None, is_inference=True,
        attack_iters=0, attack_step=1.0, attack_eps=8 / 255,
        attack='none', trades_beta=0, delta=None,
        h_utils=None): 
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """

    topK_to_consider = (1, 5, 10, 20, 100)
    max_dist = max(distances.distances.values())

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    dist_id = "ilsvrc_dist"

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

    # Affects the behaviour of components such as batch-norm
    if is_inference:
        model.eval()
    else:
        model.train()

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

            #######################################################
            # Natural training step
            #######################################################

            if attack == 'none': 

                # get model's prediction
                output = model(embeddings)

                # for soft-labels we need to add a log_softmax and get the soft labels
                loss = loss_function(output, target)

                if not is_inference:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #######################################################
            # Free adv training step
            #######################################################

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

                    # for soft-labels we need to add a log_softmax and get the soft labels
                    loss = loss_function(logits, target)

                    if not is_inference:
                        optimizer.zero_grad()
                        loss.backward()

                        with torch.no_grad():
                            grad = imgs.grad.data
                            delta += attack_step * torch.sign(grad)
                            delta = torch.clamp(delta, -attack_eps, attack_eps)
                            # delta.clamp_(-attack_eps, attack_eps)

                        optimizer.step()

            #######################################################
            # TRADES adv training step
            #######################################################

            elif attack == 'trades':
                criterion_kl = nn.KLDivLoss(size_average=False)
                model.eval()
                batch_size = embeddings.size(0)

                # with torch.no_grad():
                #     natural_dist = nn.functional.softmax(model(embeddings), dim=1).detach()

                x_adv = 0.001 * torch.randn_like(embeddings) + embeddings.detach()

                for rep in range(attack_iters):
                    x_adv.requires_grad = True

                    loss_kl = criterion_kl(nn.functional.log_softmax(model(x_adv), dim=1),
                                           nn.functional.softmax(model(embeddings), dim=1))

                    with torch.no_grad():
                        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                        x_adv = x_adv.detach() + attack_step * torch.sign(grad)
                        x_adv = torch.max(x_adv, embeddings - attack_eps)
                        x_adv = torch.min(x_adv, embeddings + attack_eps)
                        x_adv.clamp_(0.0, 1.0)

                model.train()
                x_adv.requires_grad = False

                if not is_inference:
                    optimizer.zero_grad()

                result = model(torch.cat((embeddings, x_adv), dim=0))
                output, output_adv = torch.split(result, batch_size, dim=0)

                # for soft-labels we need to add a log_softmax and get the soft labels
                natural_loss = loss_function(output, target)

                robust_loss = criterion_kl(nn.functional.log_softmax(output_adv, dim=1),
                                           nn.functional.softmax(output, dim=1)) / batch_size

                loss = natural_loss + trades_beta * robust_loss

                if not is_inference:
                    loss.backward()
                    optimizer.step()

            else:
                raise ValueError(f'{attack} attack not implemented')

            #######################################################
            # EVALUATION
            #######################################################

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


# def _generate_summary(
#         loss_accum,
#         flat_accuracy_accums,
#         hdist_accums,
#         hdist_top_accums,
#         hdist_mistakes_accums,
#         hprecision_accums,
#         hmAP_accums,
#         num_logged,
#         norm_mistakes_accum,
#         loss_id,
#         dist_id,
# ):
#     """
#     Generate dictionary with epoch's summary
#     """
#     summary = dict()
#     summary[loss_id] = loss_accum / num_logged
#     # -------------------------------------------------------------------------------------------------
#     summary.update({accuracy_ids[i]: flat_accuracy_accums[i] / num_logged for i in range(len(topK_to_consider))})
#     summary.update({dist_id + dist_avg_ids[i]: hdist_accums[i] / num_logged for i in range(len(topK_to_consider))})
#     summary.update({dist_id + dist_top_ids[i]: hdist_top_accums[i] / num_logged for i in range(len(topK_to_consider))})
#     summary.update(
#         {dist_id + dist_avg_mistakes_ids[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * topK_to_consider[i]) for i in range(len(topK_to_consider))}
#     )
#     summary.update({dist_id + hprec_ids[i]: hprecision_accums[i] / num_logged for i in range(len(topK_to_consider))})
#     summary.update({dist_id + hmAP_ids[i]: hmAP_accums[i] / num_logged for i in range(len(topK_to_consider))})
#     return summary
