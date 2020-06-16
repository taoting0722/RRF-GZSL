import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class Mapping(nn.Module):
    def __init__(self, opt):
        super(Mapping, self).__init__()
        self.latensize=opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize*2)
        self.discriminator = nn.Linear(opt.latenSize, 1)
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, train_G=False):
        laten=self.lrelu(self.encoder_linear(x))
        mus,stds = laten[:,:self.latensize],laten[:,self.latensize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))

        return mus,stds,dis_out,pred,encoder_out

class Latent_CLS(nn.Module):
    def __init__(self, opt):
        super(Latent_CLS, self).__init__()
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, latent):
        pred = self.logic(self.classifier(latent))
        return pred

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.latensize = opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize * 2)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)
    def forward(self, x,mean_mode=True):
        latent = self.lrelu(self.encoder_linear(x))
        mus, stds = latent[:, :self.latensize], latent[:, self.latensize:]
        stds = self.sigmoid(stds)
        z=reparameter(mus, stds)
        return mus, stds,z

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu



