import time
import numpy as np
import os.path
import torch
from conditional import conditional

from tqdm import tqdm
from core.model.init import hAutoAttackWrapper
from core.model.performance import accuracy

# attack imports
from autoattack import AutoAttack

topK_to_consider = (1, 5, 10, 20, 100)

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]


def eval(loader, model, loss_function, distances,
         classes, opts, epoch, prev_steps, optimizer=None, is_inference=True,
         attack_iters=0, attack_step=1.0, attack_eps=8 / 255, attack='none',
         h_utils=None, save_adv=False):
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """

    max_dist = max(distances.distances.values())
    # for each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)

    # Using different logging frequencies for training and validation
    log_freq = 1

    # strings useful for logging
    descriptor = "VAL"
    loss_id = "loss/CE"
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
    model.eval()

    time_load0 = time.time()
    batch_idx = -1

    if attack == 'NHAA':
        aa_model = hAutoAttackWrapper(model, h_utils)

    if save_adv:
        save_advs = torch.zeros(0, 3, 224, 224)
        save_true = torch.zeros(0, 3, 224, 224)

    for (embeddings, target) in tqdm(loader):
        batch_idx += 1

        this_load_time = time.time() - time_load0
        this_rest0 = time.time()

        if opts.gpu is not None:
            embeddings = embeddings.cuda(opts.gpu, non_blocking=True)
        target = target.cuda(opts.gpu, non_blocking=True)

        #######################################################
        # Compute natural evaluation
        #######################################################

        with torch.no_grad():
            output = model(embeddings)

        pred = output.max(dim=1)[1]
        correct = (pred == target)

        if save_adv:
            init_correct = correct.clone()

        x_adv = embeddings[correct, ...]
        original = embeddings.clone()

        #######################################################
        # PGD untargeted evaluation step
        #######################################################

        if attack == 'PGD':

            model.zero_grad()  # removes gradients

            x_adv = attack_eps * (2 * torch.rand_like(x_adv) - 1)  + x_adv.clone()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            for rep in range(attack_iters):

                if x_adv.size(0) == 0:
                    break

                x_adv.requires_grad = True
                output = model(x_adv)
                loss = loss_function(output, target[correct])
                loss.backward()

                with torch.no_grad():
                    x_adv += attack_step * torch.sign(x_adv.grad)
                    x_adv = torch.max(x_adv, original[correct] - attack_eps)
                    x_adv = torch.min(x_adv, original[correct] + attack_eps)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)

                adv_pred = output.max(dim=1)[1]
                adv_correct = (target[correct] == adv_pred)

                # check those that already changed its correct label
                embeddings[correct] = x_adv.detach()
                correct[correct] = adv_correct

                x_adv = embeddings[correct, ...].detach()
                model.zero_grad()

            with torch.no_grad():
                output = model(embeddings)

        #######################################################
        # hPGD untargeted evaluation step
        #######################################################

        elif attack == 'hPGD':
            model.zero_grad()  # removes gradients

            x_adv = attack_eps * (2 * torch.rand_like(x_adv) - 1)  + x_adv.clone()
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            for rep in range(attack_iters):

                if x_adv.size(0) == 0:
                    break

                x_adv.requires_grad = True
                output = model(x_adv)
                new_logits, new_target = h_utils.get_logits(output, target[correct])
                loss = loss_function(new_logits, new_target)
                loss.backward()

                with torch.no_grad():
                    x_adv += attack_step * torch.sign(x_adv.grad)
                    x_adv = torch.max(x_adv, original[correct] - attack_eps)
                    x_adv = torch.min(x_adv, original[correct] + attack_eps)
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)

                adv_pred = new_logits.max(dim=1)[1]
                adv_correct = (new_target == adv_pred)

                # check those that already changed its correct label
                embeddings[correct] = x_adv.detach()
                correct[correct] = adv_correct

                x_adv = embeddings[correct, ...].detach()
                model.zero_grad()

            with torch.no_grad():
                output = model(embeddings)

            if save_adv:
                save_advs = torch.cat((save_advs, embeddings[init_correct, ...].detach().cpu()), dim=0)
                save_true = torch.cat((save_true, original[init_correct, ...].detach().cpu()), dim=0)

        #######################################################
        # h-AutoAttack evaluation step
        #######################################################

        elif attack == 'NHAA':
            _, node_target = h_utils.get_logits(torch.zeros(embeddings.size(0), h_utils.n_classes), target)
            adversary = AutoAttack(aa_model, norm='Linf', eps=attack_eps, version='standard')
            x_adv = adversary.run_standard_evaluation(embeddings, node_target, bs=embeddings.size(0))

            with torch.no_grad():
                output = model(x_adv)

        #######################################################
        # Else...
        #######################################################

        else:
            if attack != 'none':
                raise ValueError(f'{attack} attack not implemented')

        loss = loss_function(output, target)

        # start/reset timers
        this_rest_time = time.time() - this_rest0
        time_accum += this_load_time + this_rest_time
        time_load0 = time.time()

        # only update total number of batch visited for training
        tot_steps = prev_steps

        # correct output of the classifier (for yolo-v2)
        output = corrector(output)

        # if it is time to log, compute all measures, store in summary and pass to tensorboard.
        if batch_idx % log_freq == 0:
            num_logged += 1
            # compute flat topN accuracy for N \in {topN_to_consider}
            topK_accuracies, topK_predicted_classes = accuracy(output, target, ks=topK_to_consider)
            loss_accum += loss.item()
            topK_hdist = np.empty([opts.batch_size, topK_to_consider[-1]])

            for i in range(min(opts.batch_size, embeddings.size(0))):
                for j in range(max(topK_to_consider)):
                    class_idx_ground_truth = target[i]
                    class_idx_predicted = topK_predicted_classes[i][j]
                    topK_hdist[i, j] = distances[(classes[class_idx_predicted], classes[class_idx_ground_truth])]

            # select samples which returned the incorrect class (have distance!=0 in the top1 position)
            mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
            norm_mistakes_accum += len(mistakes_ids)
            topK_hdist_mistakes = topK_hdist[mistakes_ids, :]
            # obtain similarities from distances
            topK_hsimilarity = 1 - topK_hdist / max_dist
            # all the average precisions @k \in [1:max_k]
            topK_AP = [np.sum(topK_hsimilarity[:, :k]) / np.sum(best_hier_similarities[:, :k]) for k in range(1, max(topK_to_consider) + 1)]
            for i in range(len(topK_to_consider)):
                flat_accuracy_accums[i] += topK_accuracies[i].item()
                hdist_accums[i] += np.mean(topK_hdist[:, : topK_to_consider[i]])
                hdist_top_accums[i] += np.mean([np.min(topK_hdist[b, : topK_to_consider[i]]) for b in range(opts.batch_size)])
                hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, : topK_to_consider[i]])
                hprecision_accums[i] += topK_AP[topK_to_consider[i] - 1]
                hmAP_accums[i] += np.mean(topK_AP[: topK_to_consider[i]])

    summary = _generate_summary(
        loss_accum,
        flat_accuracy_accums,
        hdist_accums,
        hdist_top_accums,
        hdist_mistakes_accums,
        hprecision_accums,
        hmAP_accums,
        num_logged,
        norm_mistakes_accum,
        loss_id,
        dist_id,
    )

    if save_adv:
        return summary, {'advs': save_advs, 'orig': save_true}
    else:
        return summary

def _make_best_hier_similarities(classes, distances, max_dist):
    """
    For each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    """
    distance_matrix = np.zeros([len(classes), len(classes)])
    best_hier_similarities = np.zeros([len(classes), len(classes)])

    for i in range(len(classes)):
        for j in range(len(classes)):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]

    for i in range(len(classes)):
        best_hier_similarities[i, :] = 1 - np.sort(distance_matrix[i, :]) / max_dist

    return best_hier_similarities


def _generate_summary(
        loss_accum,
        flat_accuracy_accums,
        hdist_accums,
        hdist_top_accums,
        hdist_mistakes_accums,
        hprecision_accums,
        hmAP_accums,
        num_logged,
        norm_mistakes_accum,
        loss_id,
        dist_id,
):
    """
    Generate dictionary with epoch's summary
    """
    summary = dict()
    summary[loss_id] = loss_accum / num_logged
    # -------------------------------------------------------------------------------------------------
    summary.update({accuracy_ids[i]: flat_accuracy_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + dist_avg_ids[i]: hdist_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + dist_top_ids[i]: hdist_top_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update(
        {dist_id + dist_avg_mistakes_ids[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * topK_to_consider[i]) for i in range(len(topK_to_consider))}
    )
    summary.update({dist_id + hprec_ids[i]: hprecision_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + hmAP_ids[i]: hmAP_accums[i] / num_logged for i in range(len(topK_to_consider))})
    return summary
