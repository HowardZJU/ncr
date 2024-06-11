import numpy as np
import sklearn
from copy import deepcopy
from auuc import auuc_score
import ot
import torch
import random



class EarlyStopping:
    """
    For avoiding nan in test. The results in PURL will be not reproducible using this plugin since the loaded state_dict is the one before 5 epochs.
    """
    def __init__(self, patience, delta, maximize=True):
        self.patience = patience
        self.delta = delta
        self.maximize = maximize

        self.loss_best = 1e-9
        self.early_stop = False
        self.counter = 0
        self.state_dict = None

    def __call__(self, loss, state_dict=None):
        update = False
        # loss *= -1 if self.maximize is True else loss
        relative_loss_improve = (loss - self.loss_best)/(self.loss_best+1e-10)
        # relative_loss_reduce *= -1 if self.maximize is True else relative_loss_reduce
        if state_dict is None:
            if self.delta < relative_loss_improve:
                print(self.loss_best, loss)
                self.loss_best = loss
                self.counter = 0
                update = True
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if np.isnan(loss) or self.counter >= self.patience:
                self.early_stop = True

            elif self.delta < relative_loss_improve:
                print(self.loss_best, loss)
                self.loss_best = loss
                self.counter = 0
                self.state_dict = state_dict
                update = True

            elif self.delta > relative_loss_improve:
                self.counter += 1
        return update


def mmd2_lin(x_0, x_1, p=0.5):
    ''' Linear MMD '''

    x_0 = x_0.mean(axis=-1)
    x_1 = x_1.mean(axis=-1)
    mmd = ((2 * p * x_1 - 2 * (1 - p) * x_0) ** 2).sum()

    return mmd


def mmd2_rbf(Xc, Xt, p=0.5, sig=1):
    """ Computes the l2-RBF MMD for X given t """

    Kcc = torch.exp(-ot.dist(Xc, Xc) / (sig ** 2))
    Kct = torch.exp(-ot.dist(Xc, Xt) / (sig ** 2))
    Ktt = torch.exp(-ot.dist(Xt, Xt) / (sig ** 2))

    m = Xc.shape[0] * 1.0
    n = Xt.shape[0] * 1.0

    mmd = ((1.0 - p) ** 2) / ((m * (m - 1.0))+1e-4) * (torch.sum(Kcc) - m)
    mmd = mmd + (p ** 2) / ((n * (n - 1.0))+1e-4) * (torch.sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * torch.sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def cal_auc(yhat_cf, y_cf):
    y_cf = y_cf.astype("int")
    return sklearn.metrics.roc_auc_score(y_cf, yhat_cf)


class StandardScaler:

    # We provide our DIY scaler operator since the treatment column is special
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-6
        self.mean[-3] = 0  # Do NOT scale the treatment column
        self.std[-3] = 1
        self.mean[-1] = 0  # Do NOT scale the counterfactual outcome column (it is just used in evaluation)
        self.std[-1] = 1

        self.mean = np.zeros_like(self.mean)
        self.std = np.ones_like(self.mean)

    def transform(self, data):
        data = (data - self.mean) / (self.std)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def reverse_y(self, yf):
        y = yf * self.std[-2] + self.mean[-2]
        return y

def seed_everything(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        print ('CUDA is available')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def metric_update(metric: dict(), metric_: dict(), epoch) -> dict():
    """
    Update the metric dict
    :param metric: self.metric in the class Estimator, each value is array
    :param metric_: output of metric() function, each value is float
    :return:
    """
    for key in metric_.keys():
        metric[key] = np.concatenate([metric[key], [metric_[key]]])
    info = "Epoch {:>3}".format(epoch)
    return metric


def metric_export(path, train_metric, eval_metric, test_metric):

    with open(path+'/run.txt', 'w') as f:
        f.write("mode,pehe,auuc,rauuc,ate,att,r2_f,r2_cf,rmse_f,rmse_cf\n")
        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'train',
            train_metric['pehe'],
            train_metric['auuc'],
            train_metric['rauuc'],
            train_metric['mae_ate'],
            train_metric['mae_att'],
            train_metric['r2_f'],
            train_metric['r2_cf'],
            train_metric['rmse_f'],
            train_metric['rmse_cf']
        ))
        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'eval',
            eval_metric['pehe'],
            eval_metric['auuc'],
            eval_metric['rauuc'],
            eval_metric['mae_ate'],
            eval_metric['mae_att'],
            eval_metric['r2_f'],
            eval_metric['r2_cf'],
            eval_metric['rmse_f'],
            eval_metric['rmse_cf']
        ))

        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'test',
            test_metric['pehe'],
            test_metric['auuc'],
            test_metric['rauuc'],
            test_metric['mae_ate'],
            test_metric['mae_att'],
            test_metric['r2_f'],
            test_metric['r2_cf'],
            test_metric['rmse_f'],
            test_metric['rmse_cf']
        ))


def metrics(
        pred_0: np.ndarray,
        pred_1: np.ndarray,
        yf: np.ndarray,
        ycf: np.ndarray,
        t: np.ndarray,
        mode) -> dict:

    assert len(pred_0.shape) == 1
    assert len(pred_1.shape) == 1
    assert len(yf.shape) == 1 and len(ycf.shape) == 1
    assert len(t.shape) == 1
    from sklearn.metrics import r2_score, mean_squared_error

    length = len(t)
    y0 = yf * (1-t) + ycf * t
    y1 = yf * t + ycf * (1-t)

    # Section: factual fitting
    yf_pred = pred_1 * t + pred_0 * (1-t)
    r2_f = r2_score(yf, yf_pred)
    rmse_f = np.sqrt(mean_squared_error(yf, yf_pred))

    # Section: counterfactual fitting
    ycf_pred = pred_0 * t + pred_1 * (1-t)
    r2_cf = r2_score(ycf, ycf_pred)
    rmse_cf = np.sqrt(mean_squared_error(ycf, ycf_pred))

    # Section: ITE estimation
    _pred_0 = deepcopy(pred_0)
    _pred_1 = deepcopy(pred_1)
    if mode == "in-sample":
        _pred_0[t == 0] = y0[t == 0]
        _pred_1[t == 1] = y1[t == 1]
    effect_pred = _pred_1 - _pred_0
    effect = y1 - y0

    # Negative effect
    effect_pred = effect_pred
    effect = effect

    pehe = np.sqrt(np.mean((effect - effect_pred) ** 2))
    ate = np.mean(effect)
    ate_pred = np.mean(effect_pred)
    att = np.mean(effect[t == 1])
    att_pred = np.mean(effect_pred[t == 1])
    mae_ate = np.abs(ate - ate_pred)
    mae_att = np.abs(att - att_pred)
    auuc = auuc_score(yf=yf, t=t, effect_pred=effect_pred)


    return {
        "mae_ate": round(mae_ate, 5),
        "mae_att": round(mae_att, 5),
        "pehe": round(pehe, 5),
        "r2_f": round(r2_f, 5),
        "rmse_f": round(rmse_f, 5),
        "r2_cf": round(r2_cf, 5),
        "rmse_cf": round(rmse_cf, 5),
        "auuc": round(auuc.iloc[0], 5),
        "rauuc": round(auuc.iloc[1], 5)
    }

def metrics_tree(
        ite_pred: np.ndarray,
        yf: np.ndarray,
        ycf: np.ndarray,
        t: np.ndarray) -> dict:
    """
    Metric calculation for causal tree-based methods
    """
    assert len(yf.shape) == 1 and len(ycf.shape) == 1
    assert len(t.shape) == 1

    y0 = yf * (1-t) + ycf * t
    y1 = yf * t + ycf * (1-t)

    r2_f, rmse_f = 0, 0
    r2_cf, rmse_cf = 0, 0
    # Section: ITE estimation
    effect = y1 - y0
    # Negative effect
    effect_pred = ite_pred
    effect = effect
    pehe = np.sqrt(np.mean((effect - effect_pred) ** 2))
    ate = np.mean(effect)
    ate_pred = np.mean(effect_pred)
    att = np.mean(effect[t == 1])
    att_pred = np.mean(effect_pred[t == 1])
    mae_ate = np.abs(ate - ate_pred)
    mae_att = np.abs(att - att_pred)
    auuc = auuc_score(yf=yf, t=t, effect_pred=effect_pred)


    return {
        "mae_ate": round(mae_ate, 5),
        "mae_att": round(mae_att, 5),
        "pehe": round(pehe, 5),
        "r2_f": round(r2_f, 5),
        "rmse_f": round(rmse_f, 5),
        "r2_cf": round(r2_cf, 5),
        "rmse_cf": round(rmse_cf, 5),
        "auuc": round(auuc[0], 5),
        "rauuc": round(auuc[1], 5)
    }

def str2lst(string):
    elements = string.split(",")
    return list(map(int, filter(bool, elements)))