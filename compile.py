import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Generator
from datasets.facade_dataset import Dataset

import numpy as np
import cv2

batchsize =1
input_channel = 3
output_channel= 3
input_height = input_width = 256



model = Generator(input_channel, output_channel).cuda()
model.load_state_dict(torch.load('./generator_G.pth'))
model.eval()
#device = torch.device("cpu")
example = torch.rand(1, 3, 256, 256)
#mode = "cpu"
#
#traced_script_module = torch.jit.trace(model, example)
#output = traced_script_module(torch.ones(1, 3, 256, 256))
#print(output.size())
#traced_script_module.save("traced_resnet_model.pt")

#torch.jit.load("traced_resnet_model.pt")

#model = torch.jit.trace(model, example.to(device))
#model.save("Net_h{}_w{}_{}.pt".format(h, w, mode))
#print("DepthNet_h{}_w{}_{}.pt is exported".format(h, w, mode))

device = torch.device("cuda")
mode = "cuda"
model = torch.jit.trace(model, example.to(device))
model.save("Net_h{}_w{}_{}.pt".format(input_height, input_width, mode))
print("DepthNet_h{}_w{}_{}.pt is exported".format(output_channel, input_height, input_width, mode))

input_x_np = np.zeros((batchsize, input_channel, input_height, input_width)).astype(np.float32)


