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

TRAIN_STEP = 20  # used for snapshot, and adjust learning rate

# Training settings
parser = argparse.ArgumentParser(description='PyTorch NI vs CI')
parser.add_argument('--dataroot', type=str, 
                    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--input_nc', type=int, default=3, 
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB', 
                    help='chooses how image are loaded. [RGB | LAB]')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--log-dir', default='/logs',
                    help='folder to output model checkpoints')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP*3, metavar='N',
                    help='number of epochs to train (default: 90)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

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

LOG_DIR = project_root + args.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

args.dataroot = os.path.join(data_root, 'imagenet10k/data/train_images_B')
train_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(args,
                   transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=True, half_constraint=True, sampler_type='PairedSampler', **kwargs)
print('The number of train data:{}'.format(len(train_loader.dataset)))

def main():

    # instantiate model and initialize weights
    model = AutoNet(input_nc=args.input_nc, ndf=6, nonlinear='relu')
    networks.print_network(model)
    networks.init_weights(model, init_type='normal')

    if args.cuda:
        model.cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = create_optimizer(model, args.lr)
    
    for epoch in range(1, args.epochs+1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, optimizer, criterion, epoch)

def train(train_loader, model, optimizer, criterion, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, target) in pbar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data_var, target_var = Variable(data), Variable(target)

        # compute output
        prediction = model(data_var)

        loss = criterion(prediction, target_var) 
        
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(
                    epoch, batch_idx * len(data_var), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0]))

    if epoch % TRAIN_STEP == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                    '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** ((epoch - 1) // TRAIN_STEP))
    print('lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()
