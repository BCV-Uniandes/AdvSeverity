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

from core.util.rand import make_deterministic
from core.util.config import load_config
from core.data.transforms import train_transforms, val_transforms
from core.model.evaluation import eval
from core.model.hierarchy_utils import HierarchyDistances
from core.model.init import init_model_on_gpu
from core.model.run_xent import run
from core.trees import load_hierarchy, get_weighting, load_distances, get_classes

MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD"]


def main_worker(opts):

    # ==========================================
    # Enables the cudnn auto-tuner to find the best algorithm to use for your hardware
    cudnn.benchmark = True
    opts.gpu = rank

    # pretty printer for cmd line options
    pp = PrettyPrinter(indent=4)

    # Setup data loaders --------------------------------------------------------------------------------------------------------------------------------------
    train_dir = os.path.join(opts.data_path, "train")
    val_dir = os.path.join(opts.data_path, "val")

    train_dataset = datasets.ImageFolder(train_dir, train_transforms())
    val_dataset = datasets.ImageFolder(val_dir, val_transforms())
    assert train_dataset.classes == val_dataset.classes

    # check that classes are loaded in the right order
    def is_sorted(x):
        return x == sorted(x)

    assert is_sorted([d[0] for d in train_dataset.class_to_idx.items()])
    assert is_sorted([d[0] for d in val_dataset.class_to_idx.items()])

    # get data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size,
                                   shuffle=True, num_workers=opts.workers,
                                   pin_memory=True, drop_last=True,
                                   sampler=train_sampler)
    val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size,
                                 shuffle=False, num_workers=opts.workers,
                                 pin_memory=True, drop_last=False)

    # Adjust the number of epochs to the size of the dataset
    num_batches = len(train_loader)
    divisor = num_batches * (opts.attack_iter) if 'free' in opts.attack else num_batches
    opts.epochs = int(round(opts.num_training_steps / divisor))

    # Load hierarchy and classes ------------------------------------------------------------------------------------------------------------------------------
    distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)
    hierarchy = load_hierarchy(opts.data_dir)

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
    model = init_model_on_gpu(opts,
                              mean=[0.454, 0.474, 0.367],
                              std=[0.237, 0.230, 0.249])

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
    steps = _load_checkpoint(opts, model, optimizer)

    # setup loss
    loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Training/evaluation -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    delta = torch.zeros(opts.batch_size, 3, 224, 224).cuda(opts.gpu, non_blocking=True) if opts.attack == 'free' else None

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

        print('#' * 79)
        print('#' * 79)
        print('#' * 79)
        print('\n\n\nRunning Attack Evaluation')
        print('Attack:', opts.evaluate)
        print('Epsilon:', opts.attack_eps)
        print('Step:', opts.attack_step)
        print('Iterations:', opts.attack_iter, '\n\n\n')

        model.eval()

        summary_val = eval(val_loader, model, loss_function, distances,
                           classes, opts, opts.start_epoch, steps,
                           None, is_inference=True,
                           attack_iters=opts.attack_iter, attack_step=opts.attack_step,
                           attack_eps=opts.attack_eps, attack=opts.evaluate,
                           h_utils=h_utils)

        with open(get_name(opts), "w") as fp:
            json.dump(summary_val, fp)

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

        # Actual training
        summary_train, steps, delta = run(train_loader, model, loss_function, distances,
                                   classes, opts, epoch, steps,
                                   optimizer, is_inference=False,
                                   attack_iters=opts.attack_iter, attack_step=opts.attack_step,
                                   attack_eps=opts.attack_eps, attack=opts.attack,
                                   trades_beta=opts.trades_beta, delta=delta,
                                   h_utils=h_utils)

        with open(os.path.join(opts.out_folder, "json/train", json_name), "w") as fp:
           json.dump(summary_train, fp)

        # print summary of the epoch and save checkpoint
        state = {"epoch": epoch + 1, "steps": steps, "arch": opts.arch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        # _save_checkpoint(state, do_validate, epoch, opts.out_folder)
        _save_checkpoint(state, False, epoch, opts.out_folder)

        # validation
        if do_validate:
            summary_val, steps, _ = run(val_loader, model, loss_function, distances,
                                        classes, opts, epoch, steps,
                                        is_inference=True, attack='none', h_utils=h_utils)

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


def _load_checkpoint(opts, model, optimizer):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")):
        print("=> loading checkpoint '{}'".format(opts.out_folder))
        checkpoint = torch.load(os.path.join(opts.out_folder, "checkpoint.pth.tar"), map_location='cpu')
        opts.start_epoch = checkpoint["epoch"]
        state_dict = checkpoint['state_dict']
        pretrained_dict = {k if k[:7] != 'module.' else k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps = checkpoint["steps"]
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.out_folder, checkpoint["epoch"]))

    else:
        steps = 0
        print("=> no checkpoint found at '{}'".format(opts.out_folder))

    return steps


def _get_current_epoch(opts):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")):
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
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=True, )
    
    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0)
    
    elif opts.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0, nesterov=False, )
    
    else:
        raise ValueError("Unknown optimizer", opts.optimizer)


def get_name(opts):

    if opts.evaluate == 'none':
        name = 'clean-eval.json'

    elif opts.evaluate == 'hPGD':
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="optimizer type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--num-training-steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data-dir", default="data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default=None, help="path to the model folder")
    
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log-freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val-freq", default=5, type=int, help="Validate every val_freq epochs (except the first 10 and last 10)")
    
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=5, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=None, type=str, help="GPU id to use.")
    
    # Evaluation Attack ---------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--evaluate", default=None, help='Evaluation protocol',
                        choices=['none', 'PGD', 'hPGD', 'NHAA'])
    parser.add_argument("--hPGD", default='NHA', type=str, help='Hierarchical attack type',
                        choices=['NHA', 'LHA', 'GHA'])
    parser.add_argument("--hPGD-level", default=3, type=int, help='hPGD height level')

    # Training/Evaluation Attack ---------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--attack-iter", default=0, type=int, help='Attack training iterations')
    parser.add_argument("--attack-step", default=0, type=float, help='Attack training step')
    parser.add_argument("--attack-eps", default=0, type=float, help='Attack training epsilon')

    # Curriculum Training -------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--curriculum-training", action='store_true', help='Use curriculum training')
    parser.add_argument("--attack", default='none', type=str, help='Adversarial training method',
                        choices=['none', 'free', 'trades'])
    parser.add_argument("--trades-beta", default=0, type=float, help='TRADES beta')
    opts = parser.parse_args()
    opts.start_epoch = 0

    if opts.gpu is not None:
        print(f'USING GPUS {opts.gpu}')
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    else:
        raise ValueError('You must use a GPU in order to train a model')

    # ==========================================
    # setup output folder
    opts.out_folder = opts.output
    if not os.path.exists(opts.out_folder):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder, "json/train"))
        os.makedirs(os.path.join(opts.out_folder, "json/val"))
        os.makedirs(os.path.join(opts.out_folder, "model_snapshots"))

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

    main_worker(opts)
