from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import myDataset

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
import numpy as np
from tqdm import tqdm
from model import AutoNet
import networks
from PIL import Image
import collections

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NI vs CI')
parser.add_argument('--dataroot', type=str, 
                    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--input_nc', type=int, default=3, 
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', 
                    help='chooses how image are loaded. [RGB | LAB]')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
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
project_root = '/home/wzquan/Project/colorization'


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

args.dataroot = os.path.join(data_root, 'test_data_all/test_images')
test_loader = torch.utils.data.DataLoader(
    myDataset.MyDataset(args,
                    transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of val data:{}'.format(len(test_loader.dataset)))

args.dataroot = os.path.join(data_root, 'test_data_all/test_images_B')
test_loader_2 = torch.utils.data.DataLoader(
    myDataset.MyDataset(args,
                    transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of val data:{}'.format(len(test_loader_2.dataset)))

args.dataroot = os.path.join(data_root, 'test_data_all/test_images_C')
test_loader_3 = torch.utils.data.DataLoader(
    myDataset.MyDataset(args,
                    transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of val data:{}'.format(len(test_loader_3.dataset)))

def main():

    # instantiate model and initialize weights
    model = AutoNet(input_nc=args.input_nc, ndf=6, nonlinear='relu')

    if args.cuda:
        model.cuda()
    
    checkpoint = torch.load(project_root + '/logs_2/checkpoint_pn_31.pth')
    model.load_state_dict(checkpoint['state_dict'])
    
    test(test_loader, model)
    test(test_loader_2, model)
    test(test_loader_3, model)

def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    oriImageLabel = []  # one dimension list, store the original label of image
    oriTestLabel = []  # one dimension list, store the predicted label of image

    correct = 0
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        
        # compute output
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        oriTestLabel.extend(pred.squeeze().cpu().numpy())
        oriImageLabel.extend(target.data.cpu().numpy())

    print(
        'Test Average Accuracy: {}/{} ({:.6f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #  Computing average accuracy
    result = np.array(oriImageLabel) == np.array(oriTestLabel)
    colorization_result = result[:len(oriImageLabel)//2]
    nature_result = result[len(oriImageLabel)//2:]

    FPR = (len(colorization_result) - colorization_result.sum())*1.0/len(colorization_result)
    FNR = (len(nature_result) - nature_result.sum())*1.0/len(nature_result)
    HTER = (FPR+FNR)*1.0/2
    print('HTER: ', 100. * HTER)

if __name__ == '__main__':
    main()
