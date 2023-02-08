# code from https://github.com/nghorbani/human_body_prior/blob/master/tutorials/ik_example_joints.py
# but adopted for this task (remove unnecessary things, vp_model loding externaly)

from typing import List, Dict

import torch
from torch import nn
import numpy as np


def ik_fit(optimizer, source_kpts_model, static_vars, vp_model, extra_params={}, on_step=None, gstep=0):
    data_loss = extra_params.get('data_loss', torch.nn.SmoothL1Loss(reduction='mean'))
    
    def fit(weights, free_vars):
        fit.gstep += 1
        optimizer.zero_grad()

        free_vars['pose_body'] = vp_model.decode(free_vars['poZ_body'])['pose_body'].contiguous().view(-1, 63)
        nonan_mask = torch.isnan(free_vars['poZ_body']).sum(-1) == 0
        res = source_kpts_model(free_vars)

        opt_objs = {}
        opt_objs['data'] = data_loss(res['source_kpts'], 
                                     static_vars['target_kpts'])
        opt_objs['poZ_body'] = \
            torch.pow(free_vars['poZ_body'][nonan_mask], 2).sum()
        opt_objs = {k: opt_objs[k] * v for k, v in weights.items() if k in opt_objs.keys()}

        loss_total = torch.sum(torch.stack(list(opt_objs.values())))
        loss_total.backward()

        fit.free_vars = {k: v for k, v in free_vars.items()}
        fit.final_loss = loss_total

        return loss_total

    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}
    # fit.nonan_mask = None
    return fit


class IK_Engine(nn.Module):

    def __init__(self,
                 vp_model,
                 data_loss,
                 optimizer_args,
                 stepwise_weights):

        super(IK_Engine, self).__init__()
        self.vp_model = vp_model
        self.data_loss = data_loss
        self.stepwise_weights = stepwise_weights
        self.optimizer_args = optimizer_args

    def forward(self, source_kpts, target_kpts, initial_body_params={}):
        bs = target_kpts.shape[0]
        comp_device = target_kpts.device

        initial_body_params['poZ_body'] = self.vp_model.encode(initial_body_params['pose_body']).mean
        free_vars = {k: torch.nn.Parameter(v.detach(), requires_grad=True) for k, v in initial_body_params.items() if
                     k in ['betas', 'trans', 'poZ_body', 'root_orient']}
        static_vars = {
            'target_kpts': target_kpts,
        }
        optimizer = torch.optim.LBFGS(list(free_vars.values()),
                                      lr=self.optimizer_args.get('lr', 1),
                                      max_iter=self.optimizer_args.get('max_iter', 100),
                                      tolerance_change=self.optimizer_args.get('tolerance_change', 1e-5),
                                      max_eval=self.optimizer_args.get('max_eval', None),
                                      history_size=self.optimizer_args.get('history_size', 100),
                                      line_search_fn='strong_wolfe')
        gstep = 0

        closure = ik_fit(optimizer,
                         source_kpts_model=source_kpts,
                         static_vars=static_vars,
                         vp_model=self.vp_model,
                         extra_params={'data_loss': self.data_loss},
                         on_step=None,
                         gstep=gstep)

        for wts in self.stepwise_weights:
            optimizer.step(lambda: closure(wts, free_vars))
            free_vars = closure.free_vars

        return closure.free_vars
