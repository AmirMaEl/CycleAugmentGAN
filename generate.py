import torch
import argparse
import os
import shutil
import cv2
import torchvision.transforms as transforms
import sys
from tqdm import tqdm
from generator_model import UNetGenerator, ResidualGenerator
from torchvision.utils import make_grid
from PIL import Image
import glob
import torch.nn as nn

'''
directory file structure:
    -Output    #output directories 
        -BRIGHT_NORMAL
        -NORMAL_BRIGHT
        -DARK_NORMAL
        _NORMAL_DARK
    -Input     #input directories
        -BRIGHT
        -NORMAL
        -DARK
'''

model_path = 'models'

parser = argparse.ArgumentParser()
parser.add_argument('-md', '--model', type=str, help='which generator model to use', default='UNet_adversarial')
parser.add_argument('-d', '--delete_input', default=False, action='store_true', help='delete the input files')
parser.add_argument('-s', '--size', type=int, default=512, help='resize the image size: tuple')
opt = parser.parse_args()


def list_generators():
    print('-' * 20)
    print('list of generators:')
    data = os.listdir(model_path)
    for item in data:
        print('  -%s' % item)


def path_to_torch_img(file):
    tmp_img = cv2.imread(file)
    tmp_img = cv2.resize(tmp_img, (opt.size, opt.size))
    tr = transforms.ToTensor()
    tmp_img = tr(tmp_img)
    tmp_img = torch.unsqueeze(tmp_img, 0)
    return tmp_img


def dir_to_model(domA, domB):
    if 'UNet' in opt.model:
        modelab = UNetGenerator(3, 3, 9, 64, nn.BatchNorm2d, False)
        modelba = UNetGenerator(3, 3, 9, 64, nn.BatchNorm2d, False)
        modelab.load_state_dict(torch.load(os.path.join(model_path, opt.model, '%s_%s.pth' % (domA, domB)),
                                           map_location='cpu'
                                           )
                                )
        modelba.load_state_dict(torch.load(os.path.join(model_path, opt.model, '%s_%s.pth' % (domB, domA)),
                                           map_location='cpu'
                                           )
                                )
        modelab.eval()
        modelba.eval()
        return modelab, modelba


def save(img, save_path, name):
    grid = make_grid(img, nrow=8, padding=2, pad_value=0, normalize=False, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    tmp_path = os.path.join(save_path, name)
    cv2.imwrite(tmp_path, ndarr)


def check_generate_dirs():
    if not os.path.isdir('Output'):
        os.mkdir('Output')
        os.mkdir(os.path.join('Output', 'BRIGHT_NORMAL'))
        os.mkdir(os.path.join('Output', 'NORMAL_BRIGHT'))
        os.mkdir(os.path.join('Output', 'NORMAL_DARK'))
        os.mkdir(os.path.join('Output', 'DARK_NORMAL'))
        print('INFO: generated Output directories')
    else:
        files = glob.glob(os.path.join('Output', '*_*', '*'), recursive=True)
        for file in files:
            os.remove(file)
        print('INFO: deleted previous output files')
    if not os.path.isdir('Input'):
        os.mkdir('Input')
        os.mkdir(os.path.join('Input', 'BRIGHT'))
        os.mkdir(os.path.join('Input', 'NORMAL'))
        os.mkdir(os.path.join('Input', 'DARK'))
        print('INFO: generated input directories')
    elif opt.delete_input:
        files = glob.glob(os.path.join('Inputs', '*', '*'), recursive=True)
        for file in files:
            os.remove(file)
        print('INFO: deleted previous input files')


def generate_and_save():
    check_generate_dirs()
    BRIGHT_in = glob.glob(os.path.join('Input', 'BRIGHT', '*'))
    NORMAL_in = glob.glob(os.path.join('Input', 'NORMAL', '*'))
    DARK_in = glob.glob(os.path.join('Input', 'DARK', '*'))

    model_nb, model_bn = dir_to_model('NORMAL', 'BRIGHT')
    model_nd, model_dn = dir_to_model('NORMAL', 'DARK')
    if len(BRIGHT_in) == 0:
        print('WARNING: no BRIGHT input files found')
    else:
        for file in tqdm(BRIGHT_in, desc='BRIGHT_NORMAL'):
            file_name = os.path.normpath(file).split(os.path.sep)[-1]
            initial_img = path_to_torch_img(file)
            bn_new = model_bn(initial_img.to('cpu'))
            save(bn_new[0].cpu(), os.path.join('Output', 'BRIGHT_NORMAL'), file_name)
    if len(NORMAL_in) == 0:
        print('WARNING: no NORMAL input files found')
    else:
        for file in tqdm(NORMAL_in, desc='NORMAL_BRIGHT'):
            file_name = os.path.normpath(file).split(os.path.sep)[-1]
            initial_img = path_to_torch_img(file)
            b_new = model_nb(initial_img.to('cpu'))
            save(b_new[0].cpu(), os.path.join('Output', 'NORMAL_BRIGHT'), file_name)
        for file in tqdm(NORMAL_in, desc='NORMAL_DARK'):
            file_name = os.path.normpath(file).split(os.path.sep)[-1]
            initial_img = path_to_torch_img(file)
            d_new = model_nd(initial_img.to('cpu'))
            save(d_new[0].cpu(), os.path.join('Output', 'NORMAL_DARK'), file_name)
    if len(DARK_in) == 0:
        print('WARNING: no DARK input files found')
    else:
        for file in tqdm(DARK_in, desc='DARK_NORMAL'):
            file_name = os.path.normpath(file).split(os.path.sep)[-1]
            initial_img = path_to_torch_img(file)
            dn_new = model_dn(initial_img.to('cpu'))
            save(dn_new[0].cpu(), os.path.join('Output', 'DARK_NORMAL'), file_name)


if __name__ == '__main__':
    if opt.model:
        generate_and_save()
