import torch
import torch.nn as nn
from copy import deepcopy
from utils import str2lst

class LinearLearner(nn.Module):  # for test

    def __init__(self, input_dim=50, params={}):

        super(LinearLearner, self).__init__()
        self.output = nn.Linear(input_dim+1, 1)

    def forward(self, x):

        output = self.output(x)
        return output


class SLearner(nn.Module):
    """
    Single learner with treatment as covariates
    """
    def __init__(self, input_dim, hparams):
        super().__init__()
        d_backbone = [input_dim + 1] + str2lst(hparams['dim_backbone'])
        d_task = [d_backbone[-1]] + str2lst(hparams['dim_task'])
        self.backbone = torch.nn.Sequential()

        for i in range(1, len(d_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(d_backbone[i-1], d_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.LeakyReLU())
            self.backbone.add_module(f"backbone_drop{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.tower = torch.nn.Sequential()
        for i in range(1, len(d_task)):
            self.tower.add_module(f"tower_dense{i}", torch.nn.Linear(d_task[i-1], d_task[i]))
            self.tower.add_module(f"tower_relu{i}", torch.nn.LeakyReLU())
            self.tower.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output = torch.nn.Sequential()
        self.output.add_module("output_dense", torch.nn.Linear(d_task[-1], 1))

        self.rep_1 = None
        self.rep_0 = None

    def forward(self, x):
        
        covariates = x[:, :-1]
        t = x[:, -1]
        covariates = torch.cat([covariates, t.reshape([-1, 1])], dim=-1)
        rep = self.backbone(covariates)
        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]
        out = self.tower(rep)
        out = self.output(out)

        return out


class TLearner(nn.Module):
    """
    Two learner with covariates in different groups modeled isolatedly.
    """
    def __init__(self, input_dim, hparams):
        super().__init__()
        d_backbone = [input_dim] + str2lst(hparams['dim_backbone'])
        d_task = [d_backbone[-1]] + str2lst(hparams['dim_task'])
        self.backbone_1 = torch.nn.Sequential()

        for i in range(1, len(d_backbone)):
            self.backbone_1.add_module(f"backbone_dense{i}", torch.nn.Linear(d_backbone[i-1], d_backbone[i]))
            self.backbone_1.add_module(f"backbone_relu{i}", torch.nn.LeakyReLU())
            self.backbone_1.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.tower_1 = torch.nn.Sequential()
        for i in range(1, len(d_task)):
            self.tower_1.add_module(f"tower_dense{i}", torch.nn.Linear(d_task[i-1], d_task[i]))
            self.tower_1.add_module(f"tower_relu{i}", torch.nn.LeakyReLU())
            # self.tower_1.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=out_dim[i]))
            self.tower_1.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output_1 = torch.nn.Sequential()
        self.output_1.add_module("output_dense", torch.nn.Linear(d_task[-1], 1))

        self.backbone_0 = deepcopy(self.backbone_1)
        self.tower_0 = deepcopy(self.tower_1)
        self.output_0 = deepcopy(self.output_1)

        self.rep_1 = None
        self.rep_0 = None

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]  # shape: (-1)
        rep_1 = self.backbone_1(covariates)
        rep_0 = self.backbone_0(covariates)
        out_1 = self.tower_1(rep_1)
        out_0 = self.tower_0(rep_0)
        out_1 = self.output_1(out_1)
        out_0 = self.output_0(out_0)

        self.rep_1 = rep_1[t == 1]
        self.rep_0 = rep_0[t == 0]

        t = t.reshape(-1, 1)
        output_f = t * out_1 + (1 - t) * out_0

        return output_f


class YLearner(nn.Module):
    """
    TARNet which combines T-learner and S-learner.
    """
    def __init__(self, input_dim, hparams):

        super().__init__()
        d_backbone = [input_dim] + str2lst(hparams['dim_backbone'])
        d_task = [d_backbone[-1]] + str2lst(hparams['dim_task'])
        self.treat_embed = hparams.get('treat_embed', True)

        self.backbone = torch.nn.Sequential()
        for i in range(1, len(d_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(d_backbone[i-1], d_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            # self.backbone.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=d_backbone[i]))
            self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        if self.treat_embed is True: # 拼接treatment带来的
            d_task[0] += 2

        self.tower_1 = torch.nn.Sequential()
        for i in range(1, len(d_task)):
            self.tower_1.add_module(f"tower_dense{i}", torch.nn.Linear(d_task[i-1], d_task[i]))
            self.tower_1.add_module(f"tower_relu{i}", torch.nn.ELU())
            # self.tower_1.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=out_dim[i]))
            self.tower_1.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output_1 = torch.nn.Sequential()
        self.output_1.add_module("output_dense", torch.nn.Linear(d_task[-1], 1))

        self.tower_0 = deepcopy(self.tower_1)
        self.output_0 = deepcopy(self.output_1)

        self.rep_1, self.rep_0 = None, None
        self.out_1, self.out_0 = None, None
        self.embedding = nn.Embedding(2, 2)

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.backbone(covariates)
        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]
        rep_t = torch.cat([rep, self.embedding(t.int())], dim=-1) if self.treat_embed is True else rep

        self.out_1 = self.output_1(self.tower_1(rep_t))
        self.out_0 = self.output_0(self.tower_0(rep_t))

        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0

        return output_f



