# coding=utf-8
import argparse
import torch.nn
from torch.utils.tensorboard import SummaryWriter
import os
import math
from models import *
from utils import *
from base_estimator import BaseEstimator

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


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

        self.tower.add_module("tower_output", torch.nn.Linear(d_task[-1], 1))

    def forward(self, x):
        
        covariates = x[:, :-1]
        t = x[:, -1]
        covariates = torch.cat([covariates, t.reshape([-1, 1])], dim=-1)
        rep = self.backbone(covariates)
        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]
        out = self.tower(rep)

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

        self.tower_1.add_module("tower_output", torch.nn.Linear(d_task[-1], 1))

        self.backbone_0 = deepcopy(self.backbone_1)
        self.tower_0 = deepcopy(self.tower_1)

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]  # shape: (-1)
        rep_1 = self.backbone_1(covariates)
        rep_0 = self.backbone_0(covariates)
        out_1 = self.tower_1(rep_1)
        out_0 = self.tower_0(rep_0)

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
        self.tower_1.add_module("tower_output", torch.nn.Linear(d_task[-1], 1))

        self.tower_0 = deepcopy(self.tower_1)
        self.embedding = nn.Embedding(2, 2)

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.backbone(covariates)
        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]
        rep_t = torch.cat([rep, self.embedding(t.int())], dim=-1) if self.treat_embed is True else rep

        self.out_1 = self.tower_1(rep_t)
        self.out_0 = self.tower_0(rep_t)

        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0

        return output_f


class Estimator(BaseEstimator):

    def __init__(self, board=None, hparams={}, path=None):
        super().__init__(board=board, hparams=hparams, path=path)

    def model_construct(self):
        MODELS = {
            "slearner": SLearner(self.x_dim, self.hparams),
            "tlearner": TLearner(self.x_dim, self.hparams),
            "ylearner": YLearner(self.x_dim, self.hparams),
        }
        
        self.model = MODELS[self.hparams['model']].to(DEVICE)

    def _fit(self, _data, wass_indicator):
        
        _xt, _t, _yf, _ = _data[:, :-2], _data[:, -3], _data[:, [-2]], _data[:, -1]  # do not use the ycf
        self.model.zero_grad()
        _pred_f = self.model(_xt)

        # Loss calculation
        _loss = ((_pred_f-_yf)**2).mean()
        _loss.backward()
        self.optimizer.step()

        # Metric update
        self.board.add_scalar('loss/loss', _loss.item(), global_step=self.iter_num)
        self.iter_num += 1
    

if __name__ == "__main__":
    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--model', type=str, default='ylearner')
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--epoch', type=int, default=400)
    hparams.add_argument('--seed', type=int, default=2)
    hparams.add_argument('--patience', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')
    hparams.add_argument('--root', type=str, default='debug0522')

    hparams.add_argument('--dim_backbone', type=str, default='16,16')
    hparams.add_argument('--dim_task', type=str, default='32,32')
    hparams.add_argument('--batchSize', type=int, default=32)
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)

    hparams.add_argument('--lambda', type=float, default=0)
    hparams.add_argument('--tune', type=int, default=0)

    hparams = vars(hparams.parse_args())
    
    if hparams['tune'] == 1:
        path = f"{hparams['root']}/{hparams['data']}/{hparams['model']}_{hparams['dim_backbone']}_{hparams['seed']}"
    else:
        path = f"{hparams['root']}/{hparams['data']}/{hparams['model']}_{hparams['seed']}"
    
    if os.path.exists(f"{path}/model.pt"):
        print(f"{path}/model.pt exists!")
    else:
        writer = SummaryWriter(path)
        seed_everything(hparams['seed'])
        estimator = Estimator(board=writer, hparams=hparams, path=path)
        estimator.fit()
        writer.close()
    