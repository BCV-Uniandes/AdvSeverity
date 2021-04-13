import argparse
import os
import json
import shutil
import numpy as np
from distutils.util import strtobool as boolean
from pprint import PrettyPrinter
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import torchvision.models as models
import torchvision.datasets as datasets

import sys
sys.path.append('..')

from better_mistakes.util.rand import make_deterministic
from better_mistakes.util.folders import get_expm_folder
from better_mistakes.util.label_embeddings import create_embedding_layer
from better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from better_mistakes.util.config import load_config
from better_mistakes.data.softmax_cascade import SoftmaxCascade
from better_mistakes.data.transforms import train_transforms, val_transforms
from better_mistakes.model.evaluation import eval
from better_mistakes.model.hierarchy_utils import HierarchyDistances
from better_mistakes.model.init import init_model_on_gpu
from better_mistakes.model.run_xent import run
# from better_mistakes.model.run_nn import run_nn
from better_mistakes.model.labels import make_all_soft_labels
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss, CosineLoss, RankingLoss, CosinePlusXentLoss, YOLOLoss
from better_mistakes.trees import load_hierarchy, get_weighting, load_distances, get_classes

MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
LOSS_NAMES = ["cross-entropy", "soft-labels", "hierarchical-cross-entropy", "cosine-distance", "ranking-loss", "cosine-plus-xent", "yolo-v2"]
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD"]
DATASET_NAMES = ["tiered-imagenet-84", "inaturalist19-84", "tiered-imagenet-224", "inaturalist19-224"]


def main_worker(rank, opts, world_size, distributed):

    # ==========================================
    # Enables the cudnn auto-tuner to find the best algorithm to use for your hardware
    cudnn.benchmark = True
    opts.gpu = rank

    # pretty printer for cmd line options
    pp = PrettyPrinter(indent=4)

    # Setup data loaders --------------------------------------------------------------------------------------------------------------------------------------
    train_dir = os.path.join(opts.data_path, "train")
    val_dir = os.path.join(opts.data_path, "val")

    train_dataset = datasets.ImageFolder(train_dir, train_transforms(opts.target_size, opts.data, augment=opts.data_augmentation, normalize=True))
    # train_dataset = datasets.ImageFolder(val_dir, val_transforms(opts.data, normalize=False))
    val_dataset = datasets.ImageFolder(val_dir, val_transforms(opts.data, normalize=False))
    assert train_dataset.classes == val_dataset.classes

    # check that classes are loaded in the right order
    def is_sorted(x):
        return x == sorted(x)

    assert is_sorted([d[0] for d in train_dataset.class_to_idx.items()])
    assert is_sorted([d[0] for d in val_dataset.class_to_idx.items()])

    if distributed:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        opts.batch_size = int(opts.batch_size / world_size)
        opts.workers = int(opts.workers / world_size)

    # data samplers for distributed training, evaluation is done on one gpu!
    train_sampler = data.distributed.DistributedSampler(train_dataset) if distributed else None

    # get data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size,
                                   shuffle=not distributed, num_workers=opts.workers,
                                   pin_memory=True, drop_last=True,
                                   sampler=train_sampler)
    val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size,
                                 shuffle=False, num_workers=opts.workers,
                                 pin_memory=True, drop_last=False)

    # Adjust the number of epochs to the size of the dataset
    num_batches = len(train_loader) * world_size
    divisor = num_batches * (opts.attack_iter) if 'free' in opts.attack else num_batches
    opts.epochs = int(round(opts.num_training_steps / divisor))

    # Load hierarchy and classes ------------------------------------------------------------------------------------------------------------------------------
    distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)
    hierarchy = load_hierarchy(opts.data, opts.data_dir)

    classes = train_dataset.classes

    opts.num_classes = len(classes)
    opts.attack_eps = opts.attack_eps / 255
    opts.attack_step = opts.attack_step / 255

    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Model, loss, optimizer ----------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    # setup model
    model = init_model_on_gpu(world_size, opts,
                              mean=[0.454, 0.474, 0.367],
                              std=[0.237, 0.230, 0.249],
                              distributed=distributed,
                              rank=rank)

    # setup hierarchical settings
    h_utils = HierarchyDistances(hierarchy, distances, train_dataset.class_to_idx,
                                 attack=opts.hPGD, level=opts.hPGD_level, topk=opts.hPGD_topk)

    if opts.curriculum_training:
        h_utils.get_curriculum(opts.epochs)
        _get_current_epoch(opts)
        current_stage = h_utils.get_current_stage(opts.start_epoch - 1)
        labels_transform = h_utils.initialize_new_classification_layer_copy(model, current_stage)
        model.cuda(opts.gpu)
        train_dataset.target_transform = labels_transform
        val_dataset.target_transform = labels_transform

    # setup optimizer
    optimizer = _select_optimizer(model, opts)

    # load from checkpoint if existing
    steps = _load_checkpoint(opts, model, optimizer, distributed)


    # setup loss
    if opts.loss == "cross-entropy":
        loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)
    elif opts.loss == "soft-labels":
        loss_function = nn.KLDivLoss().cuda(opts.gpu)
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).cuda(opts.gpu)
    elif opts.loss == "cosine-distance":
        loss_function = CosineLoss(emb_layer).cuda(opts.gpu)
    elif opts.loss == "ranking-loss":
        loss_function = RankingLoss(emb_layer, opts.batch_size, opts.devise_single_negative, margin=0.1).cuda(opts.gpu)
    elif opts.loss == "cosine-plus-xent":
        loss_function = CosinePlusXentLoss(emb_layer).cuda(opts.gpu)
    else:
        raise RuntimeError("Unkown loss {}".format(opts.loss))

    corrector = lambda x: x

    # create the solft labels
    soft_labels = make_all_soft_labels(distances, classes, opts.beta)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Training/evaluation -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    if opts.attack == 'free':
        delta = torch.zeros(opts.batch_size, 3, 224, 224).cuda(opts.gpu, non_blocking=True)
    elif opts.attack == 'h-free':
        delta = [torch.zeros(opts.batch_size, 3, 224, 224).cuda(opts.gpu, non_blocking=True),  # intra level
                 torch.zeros(opts.batch_size, 3, 224, 224).cuda(opts.gpu, non_blocking=True)]  # extra level
    else:
        delta = None

    requires_grad_to_set = True


    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Evaluation -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    if opts.evaluate is not None:

        loss_function = nn.CrossEntropyLoss(reduction="sum").cuda(opts.gpu)

        if distributed:
            raise NotImplementedError('Distributed evaluation is not implemented yet')

        print('#' * 79)
        print('#' * 79)
        print('#' * 79)
        print('\n\n\nRunning Attack Evaluation')
        print('Attack:', opts.evaluate)
        print('Epsilon:', opts.attack_eps)
        print('Step:', opts.attack_step)
        print('Iterations:', opts.attack_iter, '\n\n\n')

        model.eval()

        if opts.evaluate != 'PGD-t-all':

            # if os.path.exists(get_name(opts)):
            #     print(f'{get_name(opts)} already exists')
            #     return

            summary_val = eval(val_loader, model, loss_function, distances,
                                       soft_labels, classes, opts, opts.start_epoch, steps,
                                       None, is_inference=True, corrector=corrector,
                                       attack_iters=opts.attack_iter, attack_step=opts.attack_step,
                                       attack_eps=opts.attack_eps, attack=opts.evaluate,
                                       h_utils=h_utils)

            with open(get_name(opts), "w") as fp:
                json.dump(summary_val, fp)

        else:

            path = os.path.join(opts.out_folder,
                                "json/val/PGD-t-all-it{:d}-eps{:d}-step{:d}".format(opts.attack_iter,
                                                                                    int(255 * opts.attack_eps),
                                                                                    int(255 * opts.attack_step)))

            if os.path.exists(path):
                init_class = max(int(s.split('.')[0]) for s in os.listdir(path)) + 1
                print(f'Path {path} already exists. Initiating from class idx {init_class}')
            else:
                os.makedirs(path)
                init_class = 0

            for current_label in tqdm(range(init_class, 1010)):

                summary_val = eval(
                                         val_loader, model, loss_function, distances,
                                         soft_labels, classes, opts, opts.start_epoch, steps,
                                         None, is_inference=True, corrector=corrector,
                                         attack_iters=opts.attack_iter, attack_step=opts.attack_step,
                                         attack_eps=opts.attack_eps, attack=opts.evaluate,
                                         current_label=current_label)

                with open(os.path.join(path, str(current_label).zfill(5) + '.json'), "w") as fp:
                    json.dump(summary_val, fp)
                    fp.close()

        return

    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Training -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    for epoch in tqdm(range(opts.start_epoch, opts.epochs)):

        if opts.curriculum_training:
            stage = h_utils.get_current_stage(epoch)
            if stage != current_stage:
                current_stage = stage
                print(f'==> Setting new stage to: {stage}')
                with torch.no_grad():
                    labels_transform = h_utils.initialize_new_classification_layer_copy(model, current_stage)
                train_dataset.target_transform = labels_transform
                val_dataset.target_transform = labels_transform
                optimizer = _select_optimizer(model, opts)


        # do we validate at this epoch?
        do_validate = epoch % opts.val_freq == 0

        # name for the json file s
        json_name = "epoch.%04d.json" % epoch

        if distributed:
            dist.barrier()
            train_sampler.set_epoch(epoch)

        # Actual training
        summary_train, steps, delta = run(rank, train_loader, model, loss_function, distances,
                                   soft_labels, classes, opts, epoch, steps,
                                   optimizer, is_inference=False, corrector=corrector,
                                   attack_iters=opts.attack_iter, attack_step=opts.attack_step,
                                   attack_eps=opts.attack_eps, attack=opts.attack,
                                   trades_beta=opts.trades_beta, delta=delta,
                                   h_utils=h_utils, h_alpha=opts.h_free_alpha)

        if distributed:
            dist.barrier()

        if rank == 0:

            with open(os.path.join(opts.out_folder, "json/train", json_name), "w") as fp:
               json.dump(summary_train, fp)

            # print summary of the epoch and save checkpoint
            state = {"epoch": epoch + 1, "steps": steps, "arch": opts.arch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            # _save_checkpoint(state, do_validate, epoch, opts.out_folder)
            _save_checkpoint(state, False, epoch, opts.out_folder)

            # validation
            if do_validate:  # only validate on rank 0
                summary_val, steps, _ = run(
                    rank, val_loader, model, loss_function, distances, soft_labels, classes, opts, epoch, steps,
                    is_inference=True, corrector=corrector, attack='none', h_utils=h_utils
                )

                print("\nSummary for epoch %04d (for val set):" % epoch)
                pp.pprint(summary_val)
                print("\n\n")
                with open(os.path.join(opts.out_folder, "json/val", json_name), "w") as fp:
                    json.dump(summary_val, fp)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Custom functions ----------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


def _load_checkpoint(opts, model, optimizer, distributed):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")) and opts.pretrained_folder is None:
        print("=> loading checkpoint '{}'".format(opts.out_folder))
        checkpoint = torch.load(os.path.join(opts.out_folder, "checkpoint.pth.tar"), map_location='cpu')
        opts.start_epoch = checkpoint["epoch"]
        state_dict = checkpoint['state_dict']
        if distributed:
            pretrained_dict = {'module.' + k if k[:7] != 'module.' else k: v for k, v in state_dict.items()}
        else:
            pretrained_dict = {k if k[:7] != 'module.' else k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps = checkpoint["steps"]
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.out_folder, checkpoint["epoch"]))

    elif opts.pretrained_folder is not None:
        print("=> loading pretrained checkpoint '{}'".format(opts.pretrained_folder))
        checkpoint = torch.load(opts.pretrained_folder)
        checkpoint['state_dict'] = {'model.' + k: v for k, v in checkpoint['state_dict'].items()}  # just for loading the checkpoint
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        steps = 0
        print("=> loaded pretrained checkpoint '{}' (epoch {})".format(opts.pretrained_folder, checkpoint["epoch"]))

    else:
        steps = 0
        print("=> no checkpoint found at '{}'".format(opts.out_folder))

    return steps

def _get_current_epoch(opts):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")) and opts.pretrained_folder is None:
        checkpoint = torch.load(os.path.join(opts.out_folder, "checkpoint.pth.tar"), map_location='cpu')
        opts.start_epoch = checkpoint["epoch"]


def _save_checkpoint(state, do_validate, epoch, out_folder):
    filename = os.path.join(out_folder, "checkpoint.pth.tar")
    torch.save(state, filename)
    if do_validate:
        snapshot_name = "checkpoint.epoch%04d" % epoch + ".pth.tar"
        shutil.copy(filename, os.path.join(out_folder, "model_snapshots", snapshot_name))


def _select_optimizer(model, opts):
    if opts.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=False)
    elif opts.optimizer == "adam_amsgrad":
        if opts.devise or opts.barzdenzler:
            return torch.optim.Adam(
                [
                    {"params": model.model.conv1.parameters()},
                    {"params": model.model.layer1.parameters()},
                    {"params": model.model.layer2.parameters()},
                    {"params": model.model.layer3.parameters()},
                    {"params": model.model.layer4.parameters()},
                    {"params": model.model.fc.parameters(), "lr": opts.lr_fc, "weight_decay": opts.weight_decay_fc},
                ],
                lr=opts.lr,
                weight_decay=opts.weight_decay,
                amsgrad=True,
            )
        else:
            return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=True, )
    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0)
    elif opts.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0, nesterov=False, )
    else:
        raise ValueError("Unknown optimizer", opts.loss)


def get_name(opts):

    if opts.evaluate == 'none':
        name = 'clean-eval.json'

    elif opts.evaluate == 'hPGD-u':
        name = 'hPGD-u-{}-level{}-iter{:d}-eps{:d}-step{:d}-eval.json'.format(
                                                                              opts.hPGD if opts.hPGD != 'extra_topk' else opts.hPGD + str(opts.hPGD_topk),
                                                                              opts.hPGD_level,
                                                                              opts.attack_iter,
                                                                              int(255 * opts.attack_eps),
                                                                              int(255 * opts.attack_step))

    elif opts.evaluate == 'NHAA':
        name = 'NHAA-level{:d}-eps{:d}-eval.json'.format(opts.hPGD_level, int(255 * opts.attack_eps))

    else:
        name = '{}-iter{:d}-eps{:d}-step{:d}-eval.json'.format(opts.evaluate,
                                                               opts.attack_iter,
                                                               int(255 * opts.attack_eps),
                                                               int(255 * opts.attack_step))

    return os.path.join(opts.out_folder, "json/val", name)


def init_method(rank, world_size, opts):
    os.environ['MASTER_ADDR'] = opts.MASTER_ADDR
    os.environ['MASTER_PORT'] = opts.MASTER_PORT
    dist.init_process_group(backend='nccl', rank=rank,
                            world_size=world_size)
    main_worker(rank, opts, world_size, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--loss", default="cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="loss type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None, help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--num_training_steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    parser.add_argument("--beta", default=0, type=float, help="Softness parameter: the higher, the closer to one-hot encoding")
    parser.add_argument("--alpha", type=float, default=0, help="Decay parameter for hierarchical cross entropy.")
    # Devise/B&D ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--devise", type=boolean, default=False, help="Use DeViSe label embeddings")
    parser.add_argument("--devise_single_negative", type=boolean, default=False, help="Use one negative per samples instead of all")
    parser.add_argument("--barzdenzler", type=boolean, default=False, help="Use Barz&Denzler label embeddings")
    parser.add_argument("--train_backbone_after", default=float("inf"), type=float, help="Start training backbone too after this many steps")
    parser.add_argument("--use_2fc", default=False, type=boolean, help="Use two FC layers for Devise")
    parser.add_argument("--fc_inner_dim", default=1024, type=int, help="If use_2fc is True, their inner dimension.")
    parser.add_argument("--lr_fc", default=1e-3, type=float, help="learning rate for FC layers")
    parser.add_argument("--weight_decay_fc", default=0.0, type=float, help="weight decay of FC layers")
    parser.add_argument("--use_fc_batchnorm", default=False, type=boolean, help="Batchnorm layer in network head")
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data", default="tiered-imagenet-224", help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=224, type=int, help="Size of image input to the network (target resize after data augmentation)")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="../data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default=None, help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str, help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log_freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=5, type=int, help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=5, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=None, type=str, help="GPU id to use.")
    
    # Our args
    parser.add_argument("--MASTER-ADDR", default='127.0.0.1', type=str, help='ADDR for distributed training')
    parser.add_argument('--MASTER-PORT', default='29500', type=str, help='PORT for distributed training')
    parser.add_argument("--evaluate", default=None, help='Evaluation protocol',
                        choices=['none', 'PGD-u', 'PGD-t', 'FAB', 'PGD-t-all', 'hPGD-u', 'NHAA'])

    parser.add_argument("--curriculum-training", action='store_true', help='Use curriculum training')

    parser.add_argument("--attack-iter", default=0, type=int, help='Attack training iterations')
    parser.add_argument("--attack-step", default=0, type=float, help='Attack training step')
    parser.add_argument("--attack-eps", default=0, type=float, help='Attack training epsilon')
    parser.add_argument("--attack", default='none', type=str, help='Adversarial training method',
                        choices=['none', 'free', 'trades', 'PGD-u', 'h-free'])
    parser.add_argument("--trades-beta", default=0, type=float, help='TRADES beta')
    parser.add_argument("--h-free-alpha", default=0.5, type=float, help='h-free h-alpha parameter')
    parser.add_argument("--hPGD", default='extra_max', type=str, help='Hierarchical PGD type',
                        choices=['extra_max', 'extra_topk', 'extra_mean', 'intra', 'extra_wo_h'])
    parser.add_argument("--hPGD-level", default=3, type=int, help='hPGD level')
    parser.add_argument("--hPGD-topk", default=15, type=int, help='hPGD topk logits')
    parser
    opts = parser.parse_args()

    if opts.gpu is not None:
        print(f'USING GPUS {opts.gpu}')
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    else:
        print('USING ALL GPUS')

    # ==========================================
    # setup output folder
    opts.out_folder = opts.output if opts.output else get_expm_folder(__file__, "out", opts.expm_id)
    if not os.path.exists(opts.out_folder):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder, "json/train"))
        os.makedirs(os.path.join(opts.out_folder, "json/val"))
        os.makedirs(os.path.join(opts.out_folder, "model_snapshots"))

    # ==========================================
    # set if we want to output soft labels or one hot
    opts.soft_labels = opts.beta != 0

    # ==========================================
    # print options as dictionary and save to output
    # PrettyPrinter(indent=4).pprint(vars(opts))
    with open(os.path.join(opts.out_folder, "opts.json"), "w") as fp:
        json.dump(vars(opts), fp)

    # ==========================================
    # setup data path from config file if needed
    if opts.data_path is None:
        opts.data_paths = load_config(opts.data_paths_config)
        opts.data_path = opts.data_paths[opts.data]

    # ==========================================
    # setup random number generation
    if opts.seed is not None:
        make_deterministic(opts.seed)

    # ==========================================
    # invokes multiprocess or run the main_worker

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(init_method,
                 args=(world_size, opts),
                 nprocs=world_size)
    else:
        main_worker(0, opts, 1, False)
