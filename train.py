import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#torch.backends.cudnn.benchmark = True
from models import Generator, Discriminator
#from datasets.facade_dataset import Dataset
from datasets.my_dataset import Dataset
import numpy as np
import cv2

import argparse

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def tanh_range2uint_color(img):
    return (img * 128 + 128).astype(np.uint8)

def modelimg2cvimg(img):
    cvimg = np.array(img[0,:,:,:]).transpose(1,2,0)
    return tanh_range2uint_color(cvimg)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--iterate", default=10, type=int)
parser.add_argument("--lambda1", default=100, type=int)

args = parser.parse_args()

batchsize = 8
input_channel = 3
output_channel = 3
input_height = input_width = output_height = output_width = 256

input_data = Dataset(data_start = 1, data_end = 299)
train_len = input_data.len()

generator_G = Generator(input_channel, output_channel)
discriminator_D = Discriminator(input_channel, output_channel)
generator_G_refine = Generator(input_channel, output_channel)
discriminator_D_refine = Discriminator(input_channel, output_channel)
generator_G_f = Generator(input_channel, output_channel)
discriminator_D_f = Discriminator(input_channel, output_channel)

weights_init(generator_G)
weights_init(discriminator_D)
weights_init(generator_G_refine)
weights_init(discriminator_D_refine)
weights_init(generator_G_f)
weights_init(discriminator_D_f)

generator_G.cuda()
discriminator_D.cuda()
generator_G_refine.cuda()
discriminator_D_refine.cuda()
generator_G_f.cuda()
discriminator_D_f.cuda()

loss_L1 = nn.L1Loss().cuda()
loss_binaryCrossEntropy = nn.BCELoss().cuda()
loss_L1_refine = nn.L1Loss().cuda()
loss_binaryCrossEntropy_refine = nn.BCELoss().cuda()
loss_L1_f = nn.L1Loss().cuda()
loss_binaryCrossEntropy_f = nn.BCELoss().cuda()

optimizer_G = torch.optim.Adam(generator_G.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay= 0.00001)
optimizer_D = torch.optim.Adam(discriminator_D.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
optimizer_G_refine = torch.optim.Adam(generator_G_refine.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay= 0.00001)
optimizer_D_refine = torch.optim.Adam(discriminator_D_refine.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
optimizer_G_f = torch.optim.Adam(generator_G_f.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay= 0.00001)
optimizer_D_f = torch.optim.Adam(discriminator_D_f.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

input_x_np = np.zeros((batchsize, input_channel, input_height, input_width)).astype(np.float32)
input_real_np = np.zeros((batchsize, output_channel, output_height, output_width)).astype(np.float32)
mask_real_np = np.zeros((batchsize, output_channel, output_height, output_width)).astype(np.float32)

for epoch in range(args.epoch):
    for iterate in range(train_len):
    
        for i in range(batchsize):
            batch = input_data.get_image(iterate)
            #print(batch[0].size)
            input_x_np[i,:] = np.asarray(batch[0])
            input_real_np[i,:] = np.asarray(batch[1])
            mask_real_np[i,:] = np.asarray(batch[2])

        #cource
        input_x = Variable(torch.from_numpy(input_x_np)).cuda()
        input_real = Variable(torch.from_numpy(input_real_np)).cuda()
        mask_real = Variable(torch.from_numpy(mask_real_np)).cuda()

        out_generator_G = generator_G.forward(input_x)
        optimizer_D.zero_grad()

        negative_examples = discriminator_D.forward(input_x.detach(), out_generator_G.detach())
        positive_examples = discriminator_D.forward(input_x, input_real)
        loss_dis = 0.5 * ( loss_binaryCrossEntropy(positive_examples, Variable(torch.ones(positive_examples.size())).cuda()) \
                          +loss_binaryCrossEntropy(negative_examples, Variable(torch.zeros(negative_examples.size())).cuda()))

        loss_dis.backward(retain_graph=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        negative_examples = discriminator_D.forward(input_x, out_generator_G)

        loss_gen = loss_binaryCrossEntropy(negative_examples, Variable(torch.ones(negative_examples.size())).cuda()) \
                  +loss_L1(out_generator_G, input_real) * args.lambda1
        loss_gen.backward(retain_graph=True)
        optimizer_G.step()

        #refine
        out_generator_G_refine = generator_G_refine.forward(out_generator_G)
        optimizer_D_refine.zero_grad()

        negative_examples_refine = discriminator_D_refine.forward(out_generator_G.detach(), out_generator_G_refine.detach())
        positive_examples_refine = discriminator_D.forward(out_generator_G, input_real)
        loss_dis_refine = 0.5 * ( loss_binaryCrossEntropy_refine(positive_examples_refine, Variable(torch.ones(positive_examples_refine.size())).cuda()) \
                          +loss_binaryCrossEntropy_refine(negative_examples_refine, Variable(torch.zeros(negative_examples_refine.size())).cuda()))

        loss_dis_refine.backward(retain_graph=True)
        optimizer_D_refine.step()

        optimizer_G_refine.zero_grad()
        negative_examples_refine = discriminator_D_refine.forward(out_generator_G, out_generator_G_refine)

        loss_gen_refine = loss_binaryCrossEntropy_refine(negative_examples_refine, Variable(torch.ones(negative_examples_refine.size())).cuda()) \
                  +loss_L1_refine(out_generator_G_refine, input_real) * args.lambda1
        loss_gen_refine.backward(retain_graph=True)
        optimizer_G_refine.step()

        #fine
        out_generator_G_f = generator_G_f.forward(out_generator_G_refine)
        optimizer_D_f.zero_grad()

        negative_examples_f = discriminator_D_f.forward(out_generator_G_refine.detach(), out_generator_G_f.detach())
        positive_examples_f = discriminator_D_f.forward(out_generator_G_refine, mask_real)
        loss_dis_f = 0.5 * ( loss_binaryCrossEntropy_f(positive_examples_f, Variable(torch.ones(positive_examples_f.size())).cuda()) \
                          +loss_binaryCrossEntropy_f(negative_examples_f, Variable(torch.zeros(negative_examples_f.size())).cuda()))

        loss_dis_f.backward(retain_graph=True)
        optimizer_D_f.step()

        optimizer_G_f.zero_grad()
        negative_examples_f = discriminator_D_f.forward(out_generator_G_refine, out_generator_G_f)

        loss_gen_f = loss_binaryCrossEntropy(negative_examples_f, Variable(torch.ones(negative_examples_f.size())).cuda()) \
                  +loss_L1_f(out_generator_G_f, mask_real) * args.lambda1
        loss_gen_f.backward()
        optimizer_G_f.step()



        if iterate % args.iterate == 0:
            print ('{} [{}/{}] LossGen= {} LossDis= {} LossGen_ref= {} LossDis_ref= {} LossGen_f= {} LossDis_f= {}'.format(iterate, epoch+1, args.epoch, loss_gen.item(), loss_dis.item(), loss_gen_refine.item(), loss_dis_refine.item(), loss_gen_f.item(), loss_dis_f.item()))

             #""" MONITOR

            out_gen = out_generator_G.cpu()
            out_gen = out_gen.data.numpy()
            cvimg = modelimg2cvimg(out_gen)
            cv2.imwrite("./result_crf/trainGenImg_%d_stage1_pred.jpg"%iterate, cvimg)

            out_gen_refine = out_generator_G_refine.cpu()
            out_gen_refine = out_gen_refine.data.numpy()
            cvimg_refine = modelimg2cvimg(out_gen_refine)
            cv2.imwrite("./result_crf/trainGenImg_%d_stage2_pred.jpg"%iterate, cvimg_refine)

            out_gen_f = out_generator_G_f.cpu()
            out_gen_f = out_gen_f.data.numpy()
            cvimg_f = modelimg2cvimg(out_gen_f)
            cv2.imwrite("./result_crf/trainGenImg_%d_stage3_pred.jpg"%iterate, cvimg_f)

            out_gen_gt = input_real.cpu()
            out_gen_gt = out_gen_gt.data.numpy()
            cvimg_gt = modelimg2cvimg(out_gen_gt)
            cv2.imwrite("./result_crf/trainGenImg_%d_stage1_gt.jpg"%iterate, cvimg_gt)
            cv2.imwrite("./result_crf/trainGenImg_%d_stage2_gt.jpg" % iterate, cvimg_gt)

            out_gen_gt = mask_real.cpu()
            out_gen_gt = out_gen_gt.data.numpy()
            cvimg_gt = modelimg2cvimg(out_gen_gt)
            cv2.imwrite("./result_crf/trainGenImg_%d_stage3_gt.jpg"%iterate, cvimg_gt)
            #"""

    torch.save(generator_G.state_dict(),'./model_crf/generator_G_%d.pth'%epoch)
    torch.save(generator_G_refine.state_dict(),'./model_crf/generator_G_ref_%d.pth'%epoch)
    torch.save(generator_G_f.state_dict(),'./model_crf/generator_G_f_%d.pth'%epoch)
    torch.save(discriminator_D.state_dict(),'./model_crf/discriminator_D.pth')
    input_data.shuffle()



