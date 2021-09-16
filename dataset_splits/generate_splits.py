import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from shutil import copyfile

HOME_PATH = osp.join('C:\\Users','Guillaume','DATASETS','inaturalist-2019-fgvc6')
DATA_PATH = osp.join(HOME_PATH, 'train_val2019')
SPLITS_PATH = osp.join('splits_inat19', 'splits_inat19')
OUTPUT_DIR = osp.join(HOME_PATH, 'split')

def create_dir(dirr):
    if not osp.exists(dirr):
        os.makedirs(dirr)

for split_name in ['train', 'val', 'test']:
    split_path  = osp.join(SPLITS_PATH, split_name)
    files = glob(osp.join(split_path, '*.txt'))
    output_path = osp.join(OUTPUT_DIR, split_name)
    create_dir(output_path)
    print(f'Generating {split_name} split from {len(files)} files at '
        f'{split_path}. Files will be outputted to {output_path}')
    # each file is smthg like "nat0934.txt". Each line is the name of an image
    # the "934" is the number of the class
    for filename in tqdm(files):
        class_num = osp.basename(filename).split('.')[0][3:]
        with open(filename) as f:
            im_names = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        im_names = [x.strip() for x in im_names]
        for im_name in im_names:
            # find path for original image
            # import pdb; pdb.set_trace()
            src = glob(osp.join(DATA_PATH, '*', str(int(class_num)), im_name))[0]
            create_dir(osp.join(output_path, f'nat{class_num}'))
            dst = osp.join(output_path, f'nat{class_num}', im_name)
            copyfile(src, dst)

