# coding=utf-8
import argparse
import copy
import ot
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from models import *
from utils import *
import yaml
import time
from semireg_gromov import semireg_fgw
from sklearn.decomposition import PCA
import logging
from base_estimator import BaseEstimator

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def cal_wass(rep_0, rep_1, out_0, out_1, t, yf, device, hparams):

    dist = hparams['ot_scale'] * ot.dist(rep_0, rep_1)
    a = torch.ones(len(rep_0), device=device, dtype=torch.float32) / len(rep_0)
    b = torch.ones(len(rep_1), device=device, dtype=torch.float32) / len(rep_1)

    if hparams['gamma'] > 0:
        pred_0_cf = out_1[t == 0]  # predicted outcome for samples in control group given t == 1
        pred_1_cf = out_0[t == 1]  # predicted outcome for samples in treated group given t == 0
        yf_1 = yf[t == 1]
        yf_0 = yf[t == 0]
        dist_10 = ot.dist(pred_0_cf, yf_1)
        dist_01 = ot.dist(yf_0, pred_1_cf)
        dist += hparams['gamma'] * (dist_01 + dist_10)

    gamma = ot.unbalanced.sinkhorn_unbalanced(a, b, dist.detach(), reg=hparams.get('epsilon'), stopThr=1e-6, reg_m=hparams.get('kappa'))
    loss_wass = torch.sum(gamma * dist)
    return loss_wass


class Estimator(BaseEstimator):

    def __init__(self, board=None, hparams={}, path=None):
        super().__init__(board=board, hparams=hparams, path=path)
        self.iter_num = 1
    def model_construct(self):
        self.model = YLearner(self.x_dim, self.hparams).to(DEVICE)

    def _fit(self, _data, wass_indicator):
        
        _xt, _t, _yf, _ = _data[:, :-2], _data[:, -3], _data[:, [-2]], _data[:, -1]  # do not use the ycf
        self.model.zero_grad()
        _pred_f = self.model(_xt)

        # Loss calculation
        _loss_fit = ((_pred_f-_yf)**2).mean()
        _loss_wass = 0
        if wass_indicator: # Avoid samples coming from same group
            _loss_wass = cal_wass(
                rep_0=self.model.rep_0, rep_1=self.model.rep_1, 
                out_0=self.model.out_0, out_1=self.model.out_1, 
                t=_t, yf=_yf, device=DEVICE, hparams=self.hparams)

        _loss = _loss_fit + self.hparams['lambda'] * _loss_wass
        _loss.backward()
        self.optimizer.step()

        # Metric update
        _loss_wass = _loss_wass.item() if wass_indicator else 0
        self.board.add_scalar('loss/fit_loss', _loss_fit.item(), global_step=self.iter_num)
        self.board.add_scalar('loss/wass_loss', _loss_wass, global_step=self.iter_num)
        self.board.add_scalar('loss/total_loss', _loss.item(), global_step=self.iter_num)
        self.iter_num += 1
        


if __name__ == "__main__":

    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--model', type=str, default='escfr')
    hparams.add_argument('--device', type=str, default='cuda')
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--mapping', type=str, default='MLP')
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
    hparams.add_argument('--lambda', type=float, default=1.0, help='weight of wass_loss in loss function')
    hparams.add_argument('--ot_scale', type=float, default=0.1, help='weight of x distance. In IHDP, it should be set to 0.5-2.0 according to simulation conditions')
    hparams.add_argument('--epsilon', type=float, default=1.0, help='Entropic Regularization in sinkhorn. In IHDP, it should be set to 0.5-5.0 according to simulation conditions')
    hparams.add_argument('--kappa', type=float, default=1.0, help='weight of marginal constraint in UOT. In IHDP, it should be set to 0.1-5.0 according to simulation conditions')
    hparams.add_argument('--gamma', type=float, default=0.0005, help='weight of joint distribution alignment. In IHDP, it should be set to 0.0001-0.005 according to simulation conditions')
    hparams.add_argument('--ot_joint_bp', type=bool, default=True, help='weight of joint distribution alignment')

    hparams.add_argument('--tune', type=int, default=0)

    hparams = vars(hparams.parse_args())

    if hparams['tune'] == 1:
        path = f"{hparams['root']}/{hparams['data']}/{hparams['model']}/{hparams['lambda']}_{hparams['epsilon']}_{hparams['kappa']}_{hparams['gamma']}_{hparams['batchSize']}_{hparams['treat_weight']}_{hparams['ot_scale']}_{hparams['seed']}"
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
