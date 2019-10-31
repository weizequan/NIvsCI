from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import myDataset

from torch.autograd import Variable       
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import collections
from shutil import copytree, rmtree

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NI vs CI')
parser.add_argument('--dataroot', type=str, 
                    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--input_nc', type=int, default=3, 
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', 
                    help='chooses how image are loaded. [RGB | LAB]')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

data_root = '/home/wzquan/publicData/colorization/'

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

src_dir = os.path.join(data_root, 'imagenet10k/data/train_images_B')  # the dir of original training dataset
dst_dir_name = 'train_images_B_pn'
dst_dir = os.path.join(data_root, 'imagenet10k/data', dst_dir_name)
if os.path.exists(dst_dir):
  rmtree(dst_dir)
copytree(src_dir, dst_dir)

# just to get the image name in dataset
args.dataroot = dst_dir
image_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(args,
                   transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=100, shuffle=False, half_constraint=False, drop_last=False, **kwargs)
print('The number of train g data:{}'.format(len(image_loader.dataset)))
image_name = image_loader.dataset.images_txt
all_image_num = len(image_name)
print(all_image_num)

def main():

    for itr in np.arange(1, 5):
        construct_negative_samples(itr)
        
def construct_negative_samples(itr):
    print('Adding negative samples ...')

    # add negative sample
    pn_num = 0
    for idx in np.arange(0, all_image_num//2):
        img_nature = myDataset.pil_loader(os.path.join(args.dataroot, image_name[idx + all_image_num//2]))
        img_colorization = myDataset.pil_loader(os.path.join(args.dataroot, image_name[idx]))
        pn_sample = Image.blend(img_colorization, img_nature, 0.1*itr)  # out = image1 * (1.0 - alpha) + image2 * alpha
        new_image_name = image_name[idx][:-4] + '-' + str(itr) +'.png'
        pn_sample.save(os.path.join(args.dataroot, new_image_name))
        pn_num += 1
    print('The number of pn samples: ', pn_num)

    return pn_num

if __name__ == '__main__':
    main()
