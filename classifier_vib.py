import torch
import torch.nn as nn
import util



class CLASSIFIER:
    # train_Y is interger
    def __init__(self, map, latenSize, data_loader, _nclass, _cuda, generalized=True):
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.similarity = nn.MSELoss(reduce=False)
        # self.attribute=data_loader.attribute.cuda()
        self.nclass = _nclass
        self.input_dim = self.test_seen_feature.size(1)
        self.latent_dim = latenSize
        self.cuda = _cuda
        self.std=0.1
        self.criterion = nn.NLLLoss()
        self.l2_distance=nn.MSELoss(reduction='none')
        self.map = map
        for p in self.map.parameters():  # reset requires_grad
            p.requires_grad = False
        # setup optimizer
        self.index_in_epoch = 0
        self.epochs_completed = 0

        if generalized:
            self.attribute = data_loader.attribute.cuda()
            self.acc_seen=self.val(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            self.acc_unseen = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            self.H = 2 * self.acc_seen * self.acc_unseen / (self.acc_seen + self.acc_unseen)

            # print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.attribute = data_loader.attribute_unseen.cuda()
            self.acc = self.val(self.test_unseen_feature, util.map_label(self.test_unseen_label, self.unseenclasses), util.map_label(self.unseenclasses, self.unseenclasses))
            # print('acc=%.4f' % (self.acc))

        # test_label is integer

    def val(self, test_X, test_label, target_classes):
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest):

            if self.cuda:
                input=test_X[i].cuda().unsqueeze(0)
            else:
                input = test_X[i].unsqueeze(0)
            mus, stds,  encoder_out= self.map(input)
            single_input = mus.expand([self.nclass, -1])
            # aaaaa=torch.sum(self.l2_distance(single_input, self.attribute),1)

            predicted_label[i] = torch.argmin(torch.sum(self.l2_distance(single_input, self.attribute),1))
        acc = self.compute_per_class_acc(test_label, predicted_label,target_classes)
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, target_classes):
        acc_per_class = torch.FloatTensor(target_classes.size(0)).fill_(0)
        for iter,i in  enumerate(target_classes):
            idx = (test_label == i)
            aaaa=torch.sum(idx)
            acc_per_class[iter] = float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        return acc_per_class.mean()

    def compute_probability(self,input):
        probility=torch.FloatTensor(self.nclass).cuda()
        for iter,att in enumerate(self.attribute):
            att=att.unsqueeze(0)
            VAR2_inv=(1/self.std**2)*torch.eye(self.latent_dim).cuda()
            # aaa=torch.matmul(torch.matmul(input-att,VAR2_inv),(input-att).t())
            # aaa=(input-att)*VAR2_inv*(input-att).t()

            aaa=-torch.matmul(torch.matmul(input-att,VAR2_inv),(input-att).t())*0.5
            bbb=self.latent_dim*torch.log(torch.tensor(2*3.14*self.std))*0.5

            probility[iter]=torch.exp(aaa-bbb)

        return probility

        # for single_input in input:
        #     single_input=single_input.expand([self.nclass,-1])
        #     label=torch.argmax(torch.sum(self.attribute*input,1))


