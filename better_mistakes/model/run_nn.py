import os.path
import time

import numpy as np
import tensorboardX
import torch
from conditional import conditional

from better_mistakes.model.performance import accuracy_from_wordvecs
from better_mistakes.model.run_xent import _generate_summary, _update_tb_from_summary, _make_best_hier_similarities, topK_to_consider


def run_nn(rank, loader, model, loss_function, distances, classes, opts, epoch,
           prev_steps, embedding_layer, word2vec_mat, optimizer=None,
           is_inference=True, attack_iters=0, attack_step=1.0, attack_eps=8 / 255,
           attack='none', trades_beta=0, noise=None):
    """
    Runs training or inference routine for the DeViSe model.
    """

    max_dist = max(distances.distances.values())
    # for each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    dist_id = "ilsvrc_dist"

    # Initialise TensorBoard
    with_tb = opts.out_folder is not None and rank == 0

    if with_tb:
        tb_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_folder, "tb", descriptor.lower()))

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

    if attack == 'trades':
        criterion_dist = torch.nn.MSELoss(reduction='sum')

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        for batch_idx, (embeddings, target) in enumerate(loader):
            this_load_time = time.time() - time_load0
            this_rest0 = time.time()

            assert embeddings.size(0) == opts.batch_size, "Batch size should be constant (data loader should have drop_last=True)"
            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)

            # send targets to GPU
            ind_target = target.cuda(opts.gpu, non_blocking=True)

            #######################################################
            # Natural training step
            #######################################################

            if attack == 'none':

                # output model's prediction
                output = model(embeddings)

                # get the total loss
                loss = loss_function(output, ind_target)

                if not is_inference:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #######################################################
            # Free adv training step
            #######################################################

            elif attack == 'free': 

                for rep in range(attack_iters + 1):
                    noise.requires_grad = True
                    imgs = embeddings + noise
                    imgs.clamp_(0, 1.0)

                    # get model's prediction
                    logits = model(imgs)

                    if rep == 0:
                        output = logits.detach()

                    loss = loss_function(output, ind_target)

                    if not is_inference:
                        loss.backward()

                        with torch.no_grad():
                            step = torch.sign(noise.grad) * attack_step
                            noise = noise + step  # fsgm
                            noise.clamp_(-attack_eps, attack_eps)

                        optimizer.step()
                        optimizer.zero_grad()

            #######################################################
            # TRADES adv training step
            #######################################################

            elif attack == 'trades':

                model.eval()
                batch_size = embeddings.size(0)

                natural_dist = model(embeddings).detach()
                model.zero_grad()  # removes gradients

                x_adv = 0.001 * torch.randn_like(embeddings) + embeddings.detach()

                for rep in range(attack_iters):
                    x_adv.requires_grad = True

                    loss_kl = criterion_dist(model(x_adv), natural_dist) / batch_size

                    with torch.no_grad():
                        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                        x_adv = x_adv.detach() + attack_step * torch.sign(grad)
                        x_adv = torch.min(x_adv, embeddings - attack_eps)
                        x_adv = torch.max(x_adv, embeddings + attack_eps)
                        x_adv.clamp_(0.0, 1.0)

                model.train()
                x_adv.requires_grad = False

                output = model(embeddings)
                output_adv = model(x_adv)
                
                natural_loss = loss_function(output, ind_target)

                robust_loss = criterion_dist(output_adv, output)

                loss = natural_loss + trades_beta * robust_loss

                if not is_inference:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:
                raise ValueError(f'{attack} attack not implemented')

            # start/reset timers
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()

            # only update total number of batch visited for training
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            # if it is time to log, compute all measures, store in summary and pass to tensorboard.
            if batch_idx % log_freq == 0:
                num_logged += 1
                # compute flat topK accuracy for N \in {topK_to_consider}
                topK_accuracies, topK_predicted_classes = accuracy_from_wordvecs(output, target, word2vec_mat, ks=topK_to_consider)
                loss_accum += loss.item()
                topK_hdist = np.empty([opts.batch_size, topK_to_consider[-1]])
                for i in range(opts.batch_size):
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
                    flat_accuracy_accums[i] += topK_accuracies[i]
                    hdist_accums[i] += np.mean(topK_hdist[:, : topK_to_consider[i]])
                    hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, : topK_to_consider[i]])
                    hprecision_accums[i] += topK_AP[topK_to_consider[i] - 1]
                    hmAP_accums[i] += np.mean(topK_AP[: topK_to_consider[i]])

                # print ongoing results
                print(
                    "**%8s [Epoch %03d/%03d, Batch %05d/%05d]\t"
                    "Time: %2.1f ms | \t"
                    "Loss: %2.3f (%1.3f)\t"
                    % (descriptor, epoch, opts.epochs, batch_idx, len(loader), time_accum / (batch_idx + 1) * 1000, loss.item(), loss_accum / num_logged)
                )

                if not is_inference:
                    # update TensorBoard with the current snapshot of the epoch's summary
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
                    if with_tb:
                        _update_tb_from_summary(summary, tb_writer, tot_steps, loss_id, dist_id)

        # update TensorBoard with the total summary of the epoch
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
        if with_tb:
            _update_tb_from_summary(summary, tb_writer, tot_steps, loss_id, dist_id)

    if with_tb:
        tb_writer.close()

    return summary, tot_steps, noise
