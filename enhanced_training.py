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
from model import AutoNet
from PIL import Image
import collections
from shutil import copytree, rmtree
import networks
import shutil

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
parser.add_argument('--log-dir', default='/logs_2',
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

shutil.copy('/home/wzquan/Project/colorization/logs/checkpoint_60.pth', LOG_DIR)  # copy the trained model in the stage of normal training

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

src_dir = os.path.join(data_root, 'imagenet10k/data/train_images_B')  # the dir of original training dataset
dst_dir = os.path.join(data_root, 'imagenet10k/data/train_images_B_new')  # the new dir of original training dataset + negative samples
if os.path.exists(dst_dir):
    rmtree(dst_dir)

copytree(src_dir, dst_dir)

img_trans = transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])

args.dataroot = src_dir  #here, use src_dir just for extracting the original info of training data
image_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(args,
                   transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=False, half_constraint=False, drop_last=False, **kwargs)
print('The number of train data:{}'.format(len(image_loader.dataset)))
image_name = image_loader.dataset.images_txt
all_image_num = len(image_name)
print(all_image_num)


# store the name of colorized images
image_pn_add_flag = {}
for item in image_name[:all_image_num//2]:
    image_pn_add_flag[item] = True

args.dataroot = os.path.join(data_root, 'validation_nature_images_20k')  # natural validation dataset for final model selection
val_loader = torch.utils.data.DataLoader(
    myDataset.MyDataset(args,
                    transforms.Compose([
                       transforms.Resize((256,256), Image.BICUBIC),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('The number of val data:{}'.format(len(val_loader.dataset)))

def main():
    
    # instantiate model and initialize weights
    model = AutoNet(input_nc=args.input_nc, ndf=6, nonlinear='relu')
    networks.print_network(model)

    if args.cuda:
        model.cuda()

    print('using pretrained model')
    checkpoint = torch.load(project_root + args.log_dir + '/checkpoint_60.pth')
    model.load_state_dict(checkpoint['state_dict'])
    args.lr = args.lr * 0.001
    itr_start = 1

    nature = test_nature(val_loader, model)
    print(nature)
    threshold = max(2, nature*2)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = create_optimizer(model, args.lr)

    new_samples = []
    pn_num = []
    nature_error_itr_global = []
    for itr in np.arange(itr_start, 5):
        args.dataroot = dst_dir

        tmp = construct_negative_samples(model, new_samples, itr)
        pn_num.append(tmp)

        train_loader = myDataset.DataLoaderHalf(
        myDataset.MyDataset(args,
                       transforms.Compose([
                           transforms.Resize((256,256), Image.BICUBIC),
                           transforms.ToTensor(),
                           normalize
                       ])),
        batch_size=args.batch_size, shuffle=True, half_constraint=True, sampler_type='RandomBalancedSampler', **kwargs)
        print('The number of train data:{}'.format(len(train_loader.dataset)))
        args.epochs = 15  # after the new negative samples construct, the learning rate is constant, and epoch = 15

        train_multi(train_loader, optimizer, model, criterion, val_loader, itr, nature_error_itr_global)

    print(pn_num)
    model_selection(nature_error_itr_global, threshold)

def train_multi(train_loader, optimizer, model, criterion, val_loader, itr, nature_error_itr_global):

    for epoch in range(1, args.epochs+1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch, itr)

        train(train_loader, model, optimizer, criterion, epoch, itr)

        nature = test_nature(val_loader, model)

        if epoch > args.epochs//2:
            nature_error_itr_global.append(nature)

def train(train_loader, model, optimizer, criterion, epoch, itr):
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

    # itr == 0, the original training, and the save model step is TRAIN_STEP
    # itr == 1,2,3,4, the four times of negative samples insertion, and only store models of last 8 epoches
    if itr == 0:
        if epoch % TRAIN_STEP == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                        '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))
    else:
        # from 8 start
        if epoch > args.epochs//2:
            tmp = (args.epochs - args.epochs//2) * (itr - 1) + (epoch - args.epochs//2)  # the model index
            torch.save({'epoch': tmp,
                        'state_dict': model.state_dict()},
                        '{}/checkpoint_pn_{}.pth'.format(LOG_DIR, tmp))

def test_nature(test_loader, model):
    # switch to evaluate mode
    model.eval()

    oriImageLabel = []  # one dimension list, store the original label of image
    oriTestLabel = []  # one dimension list, store the predicted label of image

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data, target) in pbar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        
        # compute output
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        oriTestLabel.extend(pred.squeeze().cpu().numpy())
        oriImageLabel.extend(target.data.cpu().numpy())

    #  Computing error rate
    nature_result = np.array(oriImageLabel) == np.array(oriTestLabel)
    return (len(nature_result) - nature_result.sum())*100.0/len(nature_result)

def construct_negative_samples(model, new_samples, itr):
    print('Adding negative samples ...')
    # switch to evaluate mode
    model.eval()

    pn_data = torch.FloatTensor(1, 3, 256, 256)
    while len(new_samples) > 0:
        elem = new_samples.pop()
        img_colorization_new = myDataset.pil_loader(os.path.join(args.dataroot, elem))
        pn_sample_tmp = img_trans(img_colorization_new)
        pn_data[0, ...] = pn_sample_tmp
        input_data = Variable(pn_data.cuda(), volatile=True)
        output = model(input_data)
        output = F.softmax(output, dim=1)
        pn_pred = output.data[:,0].cpu().numpy()
        # new samples misclassified as nature
        original_colorization_name = elem[:-6] + '.png'
        if pn_pred < 0.5:
            image_pn_add_flag[original_colorization_name] = False  # the new sample is classified as nature

    assert(len(new_samples) == 0)

    pn_data_dir = '/home/wzquan/publicData/colorization/imagenet10k/data/train_images_B_pn'  # the dir of offline nagative samples
    # add new negative sample
    pn_num = 0
    for idx in np.arange(0, all_image_num//2):
        if image_pn_add_flag[image_name[idx]] == True:
            filename, fileextension = os.path.splitext(image_name[idx])
            new_image_name = filename + '-' + str(itr) +'.png'
            shutil.copy(os.path.join(pn_data_dir, new_image_name), os.path.join(args.dataroot, new_image_name))
            new_samples.append(new_image_name)
            pn_num += 1
    print('The number of pn samples: ', pn_num)

    return pn_num

def model_selection(nature_error_itr_global, threshold):

    nature_np = np.array(nature_error_itr_global)
    boundary = nature_np < threshold
    boundary_idx = np.where(boundary)
    print(boundary_idx[0] + 1)
    nature_idx = np.argmax(nature_np[boundary_idx[0]])
    model_idx = boundary_idx[0][nature_idx]
    print('The final model is checkpoint_pn_{}.pth'.format(model_idx+1))
    

def adjust_learning_rate(optimizer, epoch, itr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (itr - 1))
    print('lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print('lr: {}'.format(lr))

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
