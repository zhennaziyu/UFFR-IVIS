import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
clip_model, preprocess = clip.load('RN50')
clip_model.eval()

a = torch.randn(64, 3, 224, 224)
a = a.cuda()
feature = clip_model.encode_image(a)
for i in range (1000000):
    feature = clip_model.encode_image(a)
    print(feature.shape)