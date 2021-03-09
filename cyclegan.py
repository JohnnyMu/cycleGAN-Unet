import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
from generator import *
from discriminator import *
from DenseGenerator import *
from newGenerator import *
from denselyUnet import *
from denseUnetK import *
from vgg19 import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--d_lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=5, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lambda_id_vgg", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--g_type", type=float, default=1, help="1 origin 2 unet 3 unet3+")
parser.add_argument("--is_maxpooling", type=float, default=1, help="1 true 2 false")
parser.add_argument("--maxpool", type=str, default='True', help="1 true 2 false")
parser.add_argument("--dropout", type=str, default='True', help="1 true 2 false")


opt = parser.parse_args()
print(opt)
print(opt.maxpool)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("images/models/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

#vgg
if opt.lambda_id_vgg != 0:
    vgg = feature_net(model='vgg', n_classes=2)
    if cuda:
        vgg = vgg.cuda()
    vgg.load_state_dict(torch.load('../drive/MyDrive/net_  3.pth'))


input_shape = (opt.channels, opt.img_height, opt.img_width)

writer = SummaryWriter()

# Initialize generator and discriminator
if opt.g_type == 1:
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
# G_AB = GeneratorUNet3()
# G_BA = GeneratorUNet3()
if opt.g_type == 2:
    G_AB = GeneratorUNet(opt.channels)
    G_BA = GeneratorUNet(opt.channels)
if opt.g_type == 3:
    G_AB = Unet2(opt.channels, opt.channels, opt.is_maxpooling)
    G_BA = Unet2(opt.channels, opt.channels, opt.is_maxpooling)
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
# G_AB = Generator(opt.channels, 16, opt.channels)
# G_BA = Generator(opt.channels, 16, opt.channels)
# if opt.g_type == 3:
#     D_A = DiscriminatorUnet(opt.channels)
#     D_B = DiscriminatorUnet(opt.channels)
# else:
if opt.g_type == 4:
    G_AB = GeneratorUNet3(opt.channels, opt.channels)
    G_BA = GeneratorUNet3(opt.channels, opt.channels)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
if opt.g_type == 5:
    G_AB = DenseUnet(opt.channels, opt.channels, maxpool=opt.maxpool, dropout=opt.dropout)
    G_BA = DenseUnet(opt.channels, opt.channels, maxpool=opt.maxpool, dropout=opt.dropout)
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
if opt.g_type == 6:
    G_AB = GeneratorDenseNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorDenseNet(input_shape, opt.n_residual_blocks)
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
if opt.g_type == 7:
    G_AB = denselyUnet(opt.channels, opt.channels)
    G_BA = denselyUnet(opt.channels, opt.channels)
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
if opt.g_type == 8:
    G_AB = DenseNet2D(opt.channels, opt.channels, maxpool=(opt.maxpool == 'True'))
    G_BA = DenseNet2D(opt.channels, opt.channels, maxpool=(opt.maxpool == 'True'))
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights

    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
if opt.channels == 3:
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
if opt.channels == 1:
    transforms_ = [
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]

# Training data loader
dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done, batch=True):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    if batch:
        save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)
    else:
        save_image(image_grid, "images/models/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
if __name__ == '__main__':
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # Sematic loss
            if opt.lambda_id_vgg != 0:
                loss_sematic_A = criterion_identity(vgg.feature[0][11](fake_A), vgg.feature[0][11](real_A))
                loss_sematic_B = criterion_identity(vgg.feature[0][11](fake_B), vgg.feature[0][11](real_B))

                loss_sematic = (loss_sematic_A + loss_sematic_B) / 2

            # GAN loss
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            writer.add_scalar('saved_models/cycle_loss', loss_cycle.item(), (i + epoch * len(dataloader)))
            writer.add_scalar('saved_models/identity_loss', loss_identity.item(), (i + epoch * len(dataloader)))
            # Total loss
            if opt.lambda_id_vgg != 0:
                loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id_vgg * loss_sematic
            else:
                loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            writer.add_scalar('saved_models/G_loss', loss_G.item(), (i + epoch * len(dataloader)))

            loss_G.backward()
            optimizer_G.step()

            # if ((epoch <= 1) & (i < 3) )| (epoch > 4):

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2
            writer.add_scalar('saved_models/D_loss', loss_D.item(), (i + epoch * len(dataloader)))
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if opt.lambda_id_vgg != 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f, sematic: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        loss_sematic.item(),
                        time_left,
                    )
                )
            else:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                    )
                )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            sample_images(epoch, False)
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
