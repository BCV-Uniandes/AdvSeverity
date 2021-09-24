import time
import numpy as np
import os.path
import torch
from conditional import conditional

from tqdm import tqdm
from core.model.init import hAutoAttackWrapper
from core.model.labels import make_batch_onehot_labels, make_batch_soft_labels
from core.model.performance import accuracy

# attack imports
from autoattack import AutoAttack
# from advertorch.attacks import LinfFABAttack


topK_to_consider = (1, 5, 10, 20, 100)

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]


def eval(loader, model, loss_function, distances, all_soft_labels,
         classes, opts,
         attack_iters=0, attack_step=1.0, attack_eps=8 / 255,
         NHA_utils=None, GHA_utils=None, LHA_utils=None):  # hierarchy attack, args
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """

    # Affects the behaviour of components such as batch-norm
    model.eval()

    time_load0 = time.time()
    batch_idx = -1

    ori = torch.zeros(0, 3, 224, 224)
    pgd = torch.zeros(0, 3, 224, 224)
    lha = torch.zeros(0, 3, 224, 224)
    gha = torch.zeros(0, 3, 224, 224)
    nha = torch.zeros(0, 3, 224, 224)

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

        pgd_correct = correct.clone()
        lha_correct = correct.clone()
        gha_correct = correct.clone()
        nha_correct = correct.clone()

        pdg_embeddings = embeddings.clone()
        lha_embeddings = embeddings.clone()
        gha_embeddings = embeddings.clone()
        nha_embeddings = embeddings.clone()

        pgd_adv = pdg_embeddings[correct, ...].clone()
        lha_adv = lha_embeddings[correct, ...].clone()
        gha_adv = gha_embeddings[correct, ...].clone()
        nha_adv = nha_embeddings[correct, ...].clone()
        original = embeddings.clone()

        #######################################################
        # PGD untargeted evaluation step
        #######################################################

        model.zero_grad()  # removes gradients

        pgd_adv = attack_eps * (2 * torch.rand_like(pgd_adv) - 1)  + pgd_adv.clone()
        pgd_adv = torch.clamp(pgd_adv, 0.0, 1.0)

        for rep in range(attack_iters):

            if pgd_adv.size(0) == 0:
                break

            pgd_adv.requires_grad = True
            output = model(pgd_adv)
            loss = loss_function(output, target[pgd_correct])
            loss.backward()

            with torch.no_grad():
                pgd_adv += attack_step * torch.sign(pgd_adv.grad)
                pgd_adv = torch.max(pgd_adv, original[pgd_correct] - attack_eps)
                pgd_adv = torch.min(pgd_adv, original[pgd_correct] + attack_eps)
                pgd_adv = torch.clamp(pgd_adv, 0.0, 1.0)

            adv_pred = output.max(dim=1)[1]
            adv_pgd_correct = (target[pgd_correct] == adv_pred)

            # check those that already changed its correct label
            pdg_embeddings[pgd_correct] = pgd_adv.detach()
            pgd_correct[pgd_correct] = adv_pgd_correct

            pgd_adv = pdg_embeddings[pgd_correct, ...].detach()
            model.zero_grad()

        #######################################################
        # LHA untargeted evaluation step
        #######################################################

        model.zero_grad()  # removes gradients

        lha_adv = attack_eps * (2 * torch.rand_like(lha_adv) - 1)  + lha_adv.clone()
        lha_adv = torch.clamp(lha_adv, 0.0, 1.0)

        for rep in range(attack_iters):

            if lha_adv.size(0) == 0:
                break

            lha_adv.requires_grad = True
            output = model(lha_adv)
            new_logits, new_target = LHA_utils.get_logits(output, target[lha_correct])
            loss = loss_function(new_logits, new_target)
            loss.backward()

            with torch.no_grad():
                lha_adv += attack_step * torch.sign(lha_adv.grad)
                lha_adv = torch.max(lha_adv, original[lha_correct] - attack_eps)
                lha_adv = torch.min(lha_adv, original[lha_correct] + attack_eps)
                lha_adv = torch.clamp(lha_adv, 0.0, 1.0)

            adv_pred = new_logits.max(dim=1)[1]
            adv_lha_correct = (new_target == adv_pred)

            # check those that already changed its correct label
            lha_embeddings[lha_correct] = lha_adv.detach()
            lha_correct[lha_correct] = adv_lha_correct

            lha_adv = lha_embeddings[lha_correct, ...].detach()
            model.zero_grad()

        #######################################################
        # GHA untargeted evaluation step
        #######################################################

        model.zero_grad()  # removes gradients

        gha_adv = attack_eps * (2 * torch.rand_like(gha_adv) - 1)  + gha_adv.clone()
        gha_adv = torch.clamp(gha_adv, 0.0, 1.0)

        for rep in range(attack_iters):

            if gha_adv.size(0) == 0:
                break

            gha_adv.requires_grad = True
            output = model(gha_adv)
            new_logits, new_target = GHA_utils.get_logits(output, target[gha_correct])
            loss = loss_function(new_logits, new_target)
            loss.backward()

            with torch.no_grad():
                gha_adv += attack_step * torch.sign(gha_adv.grad)
                gha_adv = torch.max(gha_adv, original[gha_correct] - attack_eps)
                gha_adv = torch.min(gha_adv, original[gha_correct] + attack_eps)
                gha_adv = torch.clamp(gha_adv, 0.0, 1.0)

            adv_pred = new_logits.max(dim=1)[1]
            adv_gha_correct = (new_target == adv_pred)

            # check those that already changed its correct label
            gha_embeddings[gha_correct] = gha_adv.detach()
            gha_correct[gha_correct] = adv_gha_correct

            gha_adv = gha_embeddings[gha_correct, ...].detach()
            model.zero_grad()


        #######################################################
        # NHA untargeted evaluation step
        #######################################################

        model.zero_grad()  # removes gradients

        nha_adv = attack_eps * (2 * torch.rand_like(nha_adv) - 1)  + nha_adv.clone()
        nha_adv = torch.clamp(nha_adv, 0.0, 1.0)

        for rep in range(attack_iters):

            if nha_adv.size(0) == 0:
                break

            nha_adv.requires_grad = True
            output = model(nha_adv)
            new_logits, new_target = NHA_utils.get_logits(output, target[nha_correct])
            loss = loss_function(new_logits, new_target)
            loss.backward()

            with torch.no_grad():
                nha_adv += attack_step * torch.sign(nha_adv.grad)
                nha_adv = torch.max(nha_adv, original[nha_correct] - attack_eps)
                nha_adv = torch.min(nha_adv, original[nha_correct] + attack_eps)
                nha_adv = torch.clamp(nha_adv, 0.0, 1.0)

            adv_pred = new_logits.max(dim=1)[1]
            adv_nha_correct = (new_target == adv_pred)

            # check those that already changed its correct label
            nha_embeddings[nha_correct] = nha_adv.detach()
            nha_correct[nha_correct] = adv_nha_correct

            nha_adv = nha_embeddings[nha_correct, ...].detach()
            model.zero_grad()

        #######################################################
        # Compute the predictions to save those instances that 
        # were correctly attacked
        #######################################################

            with torch.no_grad():
                pgd_pred = model(pdg_embeddings).argmax(dim=1)
                lha_pred = model(lha_embeddings).argmax(dim=1)
                gha_pred = model(gha_embeddings).argmax(dim=1)
                nha_pred = model(nha_embeddings).argmax(dim=1)

                instances = correct * (lha_pred != pred) * (gha_pred != pred) * (nha_pred != pred)

                ori = torch.cat((ori, original[instances, ...].detach().cpu()), dim=0)
                pgd = torch.cat((pgd, pdg_embeddings[instances, ...].detach().cpu()), dim=0)
                lha = torch.cat((lha, lha_embeddings[instances, ...].detach().cpu()), dim=0)
                gha = torch.cat((gha, gha_embeddings[instances, ...].detach().cpu()), dim=0)
                nha = torch.cat((nha, nha_embeddings[instances, ...].detach().cpu()), dim=0)

    return {'orig': ori,
            'pgd': pgd,
            'lha': lha,
            'gha': gha,
            'nha': nha}
