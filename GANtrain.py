import math
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from util import clip_gradient

writer = SummaryWriter('log')
running_loss = 0
running_loss_output = 0
running_loss_distd = 0
running_loss_distd_2 = 0
running_loss_distg = 0
running_loss_distg_2 = 0

class Trainer(object):

    def __init__(self, cuda, model, dis, optimizer_model, optimizer_dis, train_loader, max_iter, snapshot, outpath, sshow, clip, test, stage):
        self.cuda = cuda
        self.model = model
        self.dis = dis
        self.optim_model = optimizer_model
        self.optim_dis = optimizer_dis
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.clip = clip
        self.test = test
        self.stage = stage

    def train_epoch(self):
        for batch_idx, (data, target1) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                data, target1 = data.cuda(), target1.cuda()

            global running_loss
            global running_loss_output
            global running_loss_distd
            global running_loss_distg
            # training generator
            self.optim_model.zero_grad()
            side3, side4, side5 = self.model(data)
            b = side3.size(0)

            ##training generator with discrimintor
            side3_d = torch.unsqueeze(side3, 1).cuda()
            side3_d = F.interpolate(side3_d, size=64, mode='bilinear', align_corners=None)
            dis_out3 = self.dis(side3_d)


            true_label = torch.ones(b, 1, 64, 64).cuda()
            loss_distg = nn.MSELoss()(dis_out3, true_label)
            loss_cond = nn.BCELoss(reduce=False)(side3, target1)
            loss_cond1 =-1*(torch.mean(loss_cond, dim=(1, 2)))
            loss_thr = (torch.max(loss_cond1)+torch.min(loss_cond1)) / 1.8
            loss_cond1[loss_cond1 <= loss_thr] = -1.2
            loss_cond1[loss_cond1 > loss_thr] = -0.5
            loss_cond1 = loss_cond1+1.5
            loss_cond2 = loss_cond1.clone()
            loss_cond2[loss_cond2 == 1] = 0
            loss_cond2[loss_cond2 == 0.5] = 1
            loss_cond2 = loss_cond2.unsqueeze(1).unsqueeze(1).unsqueeze(1).cuda()
            loss_cond2 = loss_cond2.expand(-1, -1, 64, 64).cuda()

            weight = sum(loss_cond1)
            weight = (b/weight)**0.5
            loss_cond1 = loss_cond1.unsqueeze(1).unsqueeze(1).cuda()
            loss_cond1 = loss_cond1.expand(-1, 256, 256).cuda()
            loss_output = nn.BCELoss(weight=loss_cond1.detach())(side3, target1)+ \
                          nn.BCELoss(weight=loss_cond1.detach())(side4, target1)+ \
                          nn.BCELoss(weight=loss_cond1.detach())(side5, target1)

            loss = loss_output + loss_distg*weight
            loss.backward()
            clip_gradient(self.optim_model, self.clip)
            self.optim_model.step()

            ##training dis
            self.optim_dis.zero_grad()
            side3_d = side3_d.detach()
            loss_cond2 = loss_cond2.detach()
            target1_d = torch.unsqueeze(target1, 1).cuda()
            target1_d = F.interpolate(target1_d, size=64, mode='bilinear', align_corners=None)
            noise = target1_d - side3_d
            noise_positive = F.relu(noise)
            side3_d_dis = side3_d + noise_positive*loss_cond2
            pred_real = self.dis(target1_d)
            loss_for_real = nn.MSELoss()(pred_real, true_label)
            false_label = torch.zeros(b, 1, 64, 64).cuda()
            pred_false3 = self.dis(side3_d_dis)
            loss_for_fake3 = nn.MSELoss()(pred_false3, false_label)
            loss_distd = 0.5*(loss_for_fake3 + loss_for_real)
            loss_distd.backward()
            self.optim_dis.step()

            running_loss += loss.item()
            running_loss_distd += loss_distd.item()
            running_loss_distg += loss_distg.item()
            running_loss_output += loss_output.item()


            # ------------------------------ output supervised --------------------------- #

            if iteration % self.sshow == (self.sshow-1):
                print('[%3d, %6d,total loss: %.3f, sal_output_loss: %.3f]' % (
                    self.test, iteration + 1, running_loss / self.sshow, running_loss_output / self.sshow))

                writer.add_scalar('total loss', running_loss / self.sshow, iteration)
                writer.add_scalar('salmap loss', running_loss_output / self.sshow, iteration)
                writer.add_scalar('distd', running_loss_distd / self.sshow, iteration)
                writer.add_scalar('distg', running_loss_distg / self.sshow, iteration)


                target1 = torch.unsqueeze(target1, 0).transpose(0, 1)
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                img = (data[0][0] * std[0] + mean[0]) * 0.299 + (data[0][1] * std[1] + mean[1]) * 0.587 + (
                            data[0][2] * std[2] + mean[2]) * 0.114
                img = img.unsqueeze(0).unsqueeze(0).cuda()
                target1_0 = target1[0].unsqueeze(0).cuda()
                side3_0 = side3[0].unsqueeze(0).unsqueeze(0).cuda()
                side4_0 = side4[0].unsqueeze(0).unsqueeze(0).cuda()
                side5_0 = side5[0].unsqueeze(0).unsqueeze(0).cuda()
                image = torch.cat((img, target1_0, side3_0, side4_0, side5_0), 0)
                writer.add_images('the results', image, iteration, dataformats='NCHW')

                running_loss = 0.0
                running_loss_output = 0.0
                running_loss_distd = 0.0
                running_loss_distg = 0.0

            if iteration <= 0:
                if iteration % self.snapshot == (self.snapshot-1):
                    savename = ('%s/snapshot_iter_weighted_%d.pth' % (self.outpath, iteration + 1 + self.test))
                    torch.save(self.model.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration + 1 + self.test))

            else:
                if iteration % 330 == (330 - 1):
                    savename = ('%s/snapshot_stage%d_64adNRGAN%d.pth' % (self.outpath, self.stage, self.test))
                    torch.save(self.model.state_dict(), savename)
                    print('save: (snapshot: %3d)' % self.test)

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            if epoch % 6 == 1:
                return
            else:
                self.epoch = epoch
                self.train_epoch()
                if self.iteration >= self.max_iter:
                    break