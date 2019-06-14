import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models.vgg_tiny import Conv, Fc
# from utils import progress_bar
from torch.autograd import Variable

from collections import deque
import os
import copy
import numpy as np
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
from collections import deque
import os
import cv2

from tensorboardX import SummaryWriter
from models import vggPCA

from differential_evolution import differential_evolution

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
os.system('rm tmp')


parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='ckpt', help='The target model')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--img_save_dir', default='./results', help='Save location for the adversarial samples.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

args = parser.parse_args()

if not os.path.exists(args.img_save_dir):
    os.mkdir(args.img_save_dir)
attack_img_save_dir = os.path.join(args.img_save_dir, 'PCA_model_pixels_{}_iteration_{}'.format(args.pixels, args.maxiter))
if not os.path.exists(attack_img_save_dir):
    os.mkdir(attack_img_save_dir)

def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x)/3)
        
        for pixel in pixels:
            x_pos, y_pos, r = pixel
            imgs[count, 0, x_pos, y_pos] = r/255.0
        count += 1

    return imgs


def predict_classes(xs, img, target_calss, net, minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone())
    input = imgs_perturbed.cuda()
    predictions = F.softmax(net(input), dim=1).data.cpu().numpy()[:, target_calss]

    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):
    attack_image = perturb_image(x, img.clone())
    input = attack_image.cuda()
    confidence = F.softmax(net(input), dim=1).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if (verbose):
        print ("Confidence: %.4f"%confidence[target_calss])
    if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
        return True
    

def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    # img: 1*3*W*H tensor
    # label: a number

    targeted_attack = target is not None
    target_calss = target if targeted_attack else label

    bounds = [(0,28), (0,28), (0,255)] * pixels

    popmul = max(1, popsize//len(bounds))

    predict_fn = lambda xs: predict_classes(
        xs, img, target_calss, net, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_calss, net, targeted_attack, verbose)

    inits = np.zeros([popmul*len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*3+0] = np.random.random()*28
            init[i*3+1] = np.random.random()*28
            init[i*3+2] = np.random.normal(128,127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

    attack_image = perturb_image(attack_result.x, img)
    attack_var = attack_image.cuda()
    predicted_probs = F.softmax(net(attack_var), dim=1).data.cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
        return 1, attack_result.x.astype(int), attack_image, predicted_class
    return 0, [None], None, None


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

    correct = 0
    success = 0

    for batch_idx, (input, target) in enumerate(loader):

        img_var = input.cuda()
        prior_probs = F.softmax(net(img_var), dim=1)
        _, indices = torch.max(prior_probs, 1)
        
        if target[0] != indices.data.cpu()[0]:
            continue

        correct += 1
        target = target.numpy()

        targets = [None] if not targeted else range(10)

        for target_calss in targets:
            if (targeted):
                if (target_calss == target[0]):
                    continue
            
            flag, x, attack_image, pred = attack(input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)

            success += flag
            if (targeted):
                success_rate = float(success)/(9*correct)
            else:
                success_rate = float(success)/correct

            if flag == 1:
                print("success rate: %.4f (%d/%d) [(x,y) = (%d,%d) and R=%d]"%(
                    success_rate, success, correct, x[0],x[1],x[2]))
                
                adversarial_sample = attack_image.squeeze().numpy().reshape((28,28,1))*255
                cv2.imwrite(os.path.join(attack_img_save_dir, 'id{}_label{}_pred{}.png'.format(success, target, pred)), adversarial_sample)
                cv2.imwrite(os.path.join(attack_img_save_dir, 'id{}_label{}.png'.format(success, target)), input.squeeze().numpy())
        if correct == args.samples:
            break
    return success_rate


def main():

    print("==> Loading data and model...")
    transform = transforms.Compose([
            transforms.ToTensor()])

    data_test = dsets.MNIST(root="./data/",
                            transform=transform,
                            train = False)

    testloader = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=False)

    class_names = [0,1,2,3,4,5,6,7,8,9]
    net = vggPCA.ConvPCANet()
    net.cuda()
    net.eval()
    cudnn.benchmark = True

    print("==> Starting attck...")
    with torch.no_grad():
        results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
        print("Final success rate: %.4f"%results)
    with open("./results/pca_result.txt","a",encoding="utf-8") as f:
        result_str = "Model:PCA  Pixels: {}  Iteration: {}  Final accuracy: {:.4f}\n".format(args.pixels, args.maxiter, 1-results)
        f.write(result_str)

if __name__ == '__main__':
    main()