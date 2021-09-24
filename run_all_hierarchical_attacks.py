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

    val_dataset = datasets.ImageFolder(val_dir, val_transforms(opts.data, normalize=False))

    if opts.chunks is not None:
        print('Running on', opts.chunks, 'chunks! Current chunk:', opts.chunk)
        val_dataset = Chunker(val_dataset, opts.chunks, opts.chunk)
    
    # check that classes are loaded in the right order
    def is_sorted(x):
        return x == sorted(x)

    assert is_sorted([d[0] for d in val_dataset.class_to_idx.items()])

    # data samplers for distributed training, evaluation is done on one gpu!

    val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size,
                                 shuffle=False, num_workers=opts.workers,
                                 pin_memory=True, drop_last=False)

    # Load hierarchy and classes ------------------------------------------------------------------------------------------------------------------------------
    distances = load_distances('ilsvrc', opts.data_dir)
    hierarchy = load_hierarchy(opts.data_dir)

    classes = val_dataset.classes

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
    NHA_utils = HierarchyDistances(hierarchy, distances, val_dataset.class_to_idx,
                                   attack='NHA', level=opts.hPGD_level)

    GHA_utils = HierarchyDistances(hierarchy, distances, val_dataset.class_to_idx,
                                   attack='GHA', level=opts.hPGD_level)

    LHA_utils = HierarchyDistances(hierarchy, distances, val_dataset.class_to_idx,
                                   attack='LHA', level=opts.hPGD_level)

    # load from checkpoint if existing
    _load_checkpoint(opts, model)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Evaluation -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    loss_function = nn.CrossEntropyLoss(reduction="sum").cuda(opts.gpu)

    print('#' * 79)
    print('#' * 79)
    print('#' * 79)
    print('\n\n\nRunning ALL Attacks')
    print('Epsilon:', opts.attack_eps)
    print('Step:', opts.attack_step)
    print('Iterations:', opts.attack_iter, '\n\n\n')

    model.eval()

    advs = eval(val_loader, model, loss_function, distances,
                soft_labels, classes, opts,
                attack_iters=opts.attack_iter, attack_step=opts.attack_step, attack_eps=opts.attack_eps,
                LHA_utils=LHA_utils, GHA_utils=GHA_utils, NHA_utils=NHA_utils)


    f_name = os.path.join(opts.out_folder, "json/val", f'joint_l{opts.hPGD_level}_adversaries')

    if opts.chunks is not None:
        f_name += '_c{}_{}'.format(opts.chunk, opts.chunks)

    f_name += '.pth'

    torch.save(advs, f_name)

    return


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Custom functions ----------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


def _load_checkpoint(opts, model):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")):
        print("=> loading checkpoint '{}'".format(opts.out_folder))
        checkpoint = torch.load(os.path.join(opts.out_folder, "checkpoint.pth.tar"), map_location='cpu')
        state_dict = checkpoint['state_dict']
        pretrained_dict = {k if k[:7] != 'module.' else k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(pretrained_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.out_folder, checkpoint["epoch"]))

    else:
        print("=> no checkpoint found at '{}'".format(opts.out_folder))


class Chunker():
    def __init__(self, dataset, chunks, chunk):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % chunks) == chunk]
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="loss type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight-decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--num-training-steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")

    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../data_paths.yml")
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
