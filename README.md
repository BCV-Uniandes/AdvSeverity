# A Hierarchical Assessment on Adversarial Severity

This is the official repository for the paper **A Hierarchical Assessment on Adversarial Severity**

## Setting up the environment

First, install the environment via anaconda by following:
```bash
conda env create -f environment.yml
conda activate better-mistakes
pip install -e .
```

Then, install AutoAttack and tqdm by running:
```bash
pip install git+https://github.com/fra31/auto-attack
pip install tqdm
```

## Data preparation

 * Download train+val sets of [iNaturalist'19](https://www.kaggle.com/c/inaturalist-2019-fgvc6)
 * Create the dataset by running the `generate_splits.py` script. Change the `DATA_PATH` AND `OUTPUT_DIR` to fit your specifications.
 * Resize all the images into the 224x224 format by using the `resize_images.sh` script.
 * Rename `data_paths.yml.example` and edit it to reflect the paths on your system. 


# WE ARE HERE FOR THE MOMENT, WE MIGHT CHANGE THE RUNNING CODES

## Running the code

The experiments of the papers are contained in the `experiments/` directory. Inside of your environment (or docker) run for example:
```
cd experiments
bash crossentropy_inaturalist19.sh
```

The entry points for the code are all inside of `scripts/`:
* `start_training.py` runs training and validation for all the methods (note: the code has been tested on single-gpu mode only)
* `plot_tradeoffs.py` produces the main plots of the paper given the json files produced by `start_training.py`
* `start_testing.py` runs the trained model on the test set for the epochs output by `plot_tradeoffs.py` (as in `experiment_to_best_epoch.json`).

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Commercial licenses available upon request.
