# Official Repository: A Hierarchical Assessment on Adversarial Severity

This is the official repository for the paper **A Hierarchical Assessment on Adversarial Severity**

Coming soon. We are creating the official github page so you can run it without any problem.

![Hierarchical Adversarial Attacks](images/h-attacks.png)

## Setting up the environment

First, install the environment via anaconda by following:
```bash
conda env create -f environment.yml
conda activate AdvSeverity
```

Then, install AutoAttack and tqdm by running:
```bash
pip install git+https://github.com/fra31/auto-attack
pip install tqdm
```

## Data preparation

 * Download train+val sets of [iNaturalist'19](https://www.kaggle.com/c/inaturalist-2019-fgvc6)
 * Unzip the zip file `dataset_splits/splits_inat19.py`
 * Create the dataset by running the `generate_splits.py` script. Change the `DATA_PATH` AND `OUTPUT_DIR` variables to fit your specifications.
 * Resize all the images into the 224x224 format by using the `resize_images.sh` script: `bash ./resize_images.sh new_splits_backup new_splits_downsampled 224x224!`
 * Rename `data_paths.yml` and edit it to reflect the paths on your system. 


## Model Downloads

To use our trained models, you must download them first. If you check the models folders, the uploaded files weight \~4kb each. Once the environment is installed and activated, run `git lfs install --local` and `git lfs pull`. Finally, check the weights of each file, it should be \~164Mb.

# Training and Evaluation

## Training

To run the training rutine, run the `main.py` script as follows:

```bash
python main.py \
    --arch resnet18 \
    --dropout 0.5 \
    --output PATH/TO/OUTPUT \
    --num-training-steps 200000 \
    --gpu GPU \
    --val-freq 1 \
    --attack-eps EPSILON \
    --attack-iter-training ITERATIONS \
    --attack-step STEP \
    --attack free \
    --curriculum-training
```

If you want to train the method without the proposed curriculum, don't use the flag `--curriculum-training`. Further, if you want to train the model with TRADES change the flag `--attack free` with `--attack trades`.

To restore the training from a checkpoint, just use the same command the fristly used.

## Evaluation

To perform the evaluation with the proposed attacks, pgd or NHAA, run the `main.py` script as follows:

```bash
python main.py \
    --arch resnet18 \
    --output PATH/TO/OUTPUT \
    --gpu GPU \
    --val-freq 1 \
    --attack-eps EPSILON \
    --attack-iter-training ITERATIONS \
    --attack-step 1 \
    --evaluate ATTACK \
    --attack-iter-training ATTACKITERATIONS
```

If `--evaluate` uses as input `hPGD`, use the `--hPGD` flag to select between `LHA`, `GHA` or `NHA` and `--hPGD-level` to select the target height. To set the total number of attack iterations, use the flag `--attack-iter-training`.

# Citation

If you found our paper or code useful, please cite our work:

```
@misc{jeanneret2021hierarchical,
      title={A Hierarchical Assessment of Adversarial Severity}, 
      author={Guillaume Jeanneret and Juan C Perez and Pablo Arbelaez},
      year={2021},
      eprint={2108.11785},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

This code was based on Bertinetto's *Making Better Mistakes* [official repository](https://github.com/fiveai/making-better-mistakes)

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Commercial licenses available upon request.

