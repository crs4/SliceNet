import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
import math


def x2image(x):
    img = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)

    return img

def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net






