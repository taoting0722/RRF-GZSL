from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util
import classifier
from center_loss import TripCenterLoss_min_margin,TripCenterLoss_margin
import classifier_latent
import sys
import model
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--dataroot', default='/home/hanzy/datasets', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=400, help='number features to generate per class')
parser.add_argument('--gzsl',action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true',  default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--latenSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--i_c', type=float, default=0.1, help='information constrain')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=3483,help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
parser.add_argument('--lr_dec', action='store_true', default=False, help='enable lr decay or not')
parser.add_argument('--lr_dec_ep', type=int, default=1, help='lr decay for every 100 epoch')
parser.add_argument('--lr_dec_rate', type=float, default=0.95, help='lr decay rate')
parser.add_argument('--final_classifier', default='softmax', help='the classifier for final classification. softmax or knn')
parser.add_argument('--k', type=int, default=1, help='k for knn')
parser.add_argument('--center_margin', type=float, default=190, help='the margin in the center loss')
parser.add_argument('--center_weight', type=float, default=0.1, help='the weight for the center loss')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
mapping= model.Mapping(opt)

cls_criterion = nn.NLLLoss()
if opt.dataset in ['CUB','SUN']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, use_gpu=opt.cuda)
elif opt.dataset in ['AWA1','FLO']:
    center_criterion = TripCenterLoss_min_margin(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, use_gpu=opt.cuda)
else:
    raise ValueError('Dataset %s is not supported'%(opt.dataset))

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)
beta=0

# i_c=0.2

if opt.cuda:
    mapping.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = netG(syn_noise,syn_att)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(mapping.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizer_center=optim.Adam(center_criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

def compute_per_class_acc_gzsl( test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx)==0:
            acc_per_class +=0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _,_,disc_interpolates,_ ,_= netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

    MI_loss = (torch.mean(kl_divergence) - i_c)

    return MI_loss

def optimize_beta(beta, MI_loss,alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))

    # return the updated beta value:
    return beta_new

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100)

for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

for epoch in range(opt.nepoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):

        for p in mapping.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            mapping.zero_grad()

            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            muR,varR,criticD_real,latent_pred,_ = mapping(input_resv)

            # latent_pred_loss=cls_criterion(latent_pred, input_label)
            criticD_real = criticD_real.mean()
            # criticD_real.backward()

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            muF, varF, criticD_fake, _,_ = mapping(fake.detach())

            criticD_fake = criticD_fake.mean()
            # criticD_fake.backward(one)
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(mapping, input_resv, fake.data)

            mi_loss=MI_loss(torch.cat((muR, muF), dim=0),torch.cat((varR, varF), dim=0), opt.i_c)

            center_loss=center_criterion(muR, input_label,margin=opt.center_margin)


            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty+0.001*criticD_real**2+beta*mi_loss+center_loss*opt.center_weight
            D_cost.backward()

            optimizerD.step()

            beta=optimize_beta(beta,mi_loss.item())

            # for param in center_criterion.parameters():
            #     param.grad.data *= (1. / args.weight_cent)
            optimizer_center.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in mapping.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        _,_,criticG_fake,latent_pred_fake ,_= mapping(fake, train_G=True)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # center_loss_f = center_criterion(z_F, input_label)

        # classification loss
        # _, _, _, , _ = mapping(fake,input_attv, mean_mode=True)

        # c_errG_latent = cls_criterion(latent_pred_fake, input_label)
        # c_errG = cls_criterion(fake, input_label)
        c_errG_fake = cls_criterion(pretrain_cls.model(fake), input_label)

        errG = G_cost + opt.cls_weight*(c_errG_fake)#+center_loss_f
        errG.backward()
        optimizerG.step()

    if opt.lr_dec:
        if (epoch + 1) % opt.lr_dec_ep == 0:
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizer_center.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    mean_lossG /=  data.ntrain / opt.batch_size
    mean_lossD /=  data.ntrain / opt.batch_size
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG_fake:%.4f,mi_loss:%.4f,beta:%.4f,center_loss:%.4f'
              % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),c_errG_fake.item(),mi_loss.item(),beta,center_loss))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    mapping.eval()

    # Generalized zero-shot learning
    # Generalized zero-shot learning
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    if opt.final_classifier == 'softmax':
        nclass = opt.nclass_all
        cls = classifier_latent.CLASSIFIER(mapping, opt.latenSize, train_X, train_Y, data, nclass, opt.cuda,
                                           opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))

    elif opt.final_classifier == 'knn':
        if epoch % 25 == 0:  ## training a knn classifier takes too much time
            clf = KNeighborsClassifier(n_neighbors=opt.k)
            train_z, _, _, _, _ = mapping(train_X.cuda())
            clf.fit(X=train_z.cpu(), y=train_Y)

            test_z_seen, _, _, _, _ = mapping(data.test_seen_feature.cuda())
            pred_Y_s = torch.from_numpy(clf.predict(test_z_seen.cpu()))
            test_z_unseen, _, _, _, _ = mapping(data.test_unseen_feature.cuda())
            pred_Y_u = torch.from_numpy(clf.predict(test_z_unseen.cpu()))
            acc_seen = compute_per_class_acc_gzsl(pred_Y_s, data.test_seen_label, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(pred_Y_u, data.test_unseen_label, data.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H))
    else:
        raise ValueError('Classifier %s is not supported' % (opt.final_classifier))
    netG.train()
    mapping.train()

