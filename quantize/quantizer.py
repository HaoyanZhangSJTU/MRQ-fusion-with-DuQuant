import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import math
from utils import get_rot, exchange_row_col, get_hadamard
from quantize.const import CLIPMAX, CLIPMIN
import random


import numpy as np
import matplotlib.pyplot as plt
import datetime

import os


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    quant_grid = torch.tensor([0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).to(x.device).to(x.dtype)

    # quant_grid = torch.tensor([0.0,  1.0,  2.0,  3.0,  4.0, 5.0,  6.0, 7.0,
    #                                   -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]).to(x.device).to(x.dtype) * 6.0 / 7.0

    labels = (x.unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    x_deq = quant_grid[labels]
    # return (x.round() - x).detach() + x
    
    
    
    return x_deq
    



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        swc=None,
        lac=None,
        act_group_size=None,
        quant_method=None,
        block_size=128,
        rotate=True,
        max_rotation_step=1024,
        permutation_times=1,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = -6.0
        self.qmax = 6.0
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rotate = rotate
        self.max_rotation_step = max_rotation_step
        self.quant_method = quant_method
        

        init_value = 4.             # init value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1*(shape[-1]//32),1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1*(shape[-1]//32),1))*init_value)
        
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        self.is_weight = shape != None
        self.permutation_times = permutation_times
        self.recorded_x_max = None
        self.let_s = None
        self.act_group_size = act_group_size
        self.lac = lac
        self.swc = swc

        self.init_duquant_params = torch.tensor(1)

        if block_size == -1:
            self.block_size = 4096
        else:
            self.block_size = block_size

        if self.rotate is None:
            self.H = get_hadamard(self.block_size)
        elif self.quant_method == 'duquant':
            self.R, self.permutation_list = [], []
            if self.rotate is not False:
                self.init_duquant_params = torch.tensor(0)
                
        self.layer_name = None
        self.draw = False
        
        
        
        self.w = True
        self.a = True
        
        self.inp_for_w = None
        self.w_for_a = None

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = -6.0
        self.qmax = 6.0

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        # exit()
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
            
        # print(self.group_size, flush=True)
        # exit()
        
        org_shape = x.shape
        x_uq = x.reshape(-1, 32)            
        
            
        x_int = round_ste(x_uq.float() / scale).half()    # avoid overflow
        
        # if round_zero_point is not None:
        #     # print(round_zero_point)
        #     x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax) 

        x_dequant = x_int
        # if round_zero_point is not None:
        #     x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant.reshape(org_shape)
    
    def permutation_random(self, weight, other=None):
        hidden_dim = weight.shape[-1]
        _mean = {}
        _weight = weight.detach().clone().abs()
        for _ in range(hidden_dim):
            _mean[_] = torch.max(_weight[:, _]).item()
        _mean = sorted(_mean.items(), key=lambda x: x[1], reverse=True)
        top_k = weight.shape[1] // self.block_size
        top_k_channel = []
        paired_list = []


        l = list(set(range(weight.shape[1])))
        random.shuffle(l)
        top_k_channel = top_k_channel + l[len(l)//2 :]
        paired_list = paired_list + l[:len(l)//2]

        top_k_channel = torch.tensor(top_k_channel)
        paired_list = torch.tensor(paired_list)

        ans = []
        top_k_channel, paired_list = top_k_channel.tolist(), paired_list.tolist()
        for i in range(hidden_dim):
            if i in top_k_channel:
                ans.append(paired_list[top_k_channel.index(i)])
            else:
                ans.append(top_k_channel[paired_list.index(i)])
        weight = weight[:, ans]
        return weight, torch.tensor(ans)
 
    def permutation_zigzag(self, weight):
        weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        weight_max = weight.abs().max(dim=0).values
        weight_mean = weight_max.mean().item()
        pairs = [(i, weight_max[i].item()) for i in range(hidden_dim)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        def zigzag(numbers):
            cur = 0
            up = True
            l = [[] for i in range(hidden_dim // self.block_size)]
            for i in range(len(numbers)):
                l[cur].append(numbers[i])
                if up:
                    cur += 1
                    if cur == len(l):
                        cur -= 1
                        up = False
                else:
                    cur -= 1
                    if cur == -1:
                        cur += 1
                        up = True
            return l
        pairs = zigzag(pairs)

        for i in range(len(pairs)):
            pairs[i].sort(key=lambda x: x[1], reverse=True)

        perm = torch.zeros(hidden_dim, dtype=torch.long)
        for i in range(len(pairs)):
            perm[i * self.block_size:(i+1) * self.block_size] = torch.tensor([_[0] for _ in pairs[i]])
        weight = weight[:, perm]
        return weight, perm

    def rotation(self, weight, max_rotation_step=None, other=None, score_func=None):
        if max_rotation_step is None:
            max_rotation_step = self.max_rotation_step
        weight = weight.detach().clone()
        _weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        exchange_ids = []
        peak_values = []
        weight = weight.reshape(-1, self.block_size)
        if other is not None:
            _other = other.detach().clone()
            other = other.reshape(-1, self.block_size)


        Rot = get_rot(self.block_size, weight.device)
        for j in range(max_rotation_step):
            if score_func is not None:
                weight_max = weight.abs().max(dim=0).values
                other_max = other.abs().max(dim=0).values
                r = score_func(weight_max, other_max).argmax().item()
                peak_values.append(score_func(weight_max[r], other_max[r]).item())
                
            else:
                r, c = divmod(weight.argmax().item(), weight.shape[-1])
                r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
                peak_values.append((weight[r, c] - weight[r2, c2]).item())
            exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
            exchange_ids.append(exchange_id)
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(weight)
            weight = torch.matmul(weight, R)
            if other is not None:
                other = torch.matmul(other, R)

        if score_func is not None:
            weight_max = weight.abs().max(dim=0).values
            other_max = other.abs().max(dim=0).values
            r = score_func(weight_max, other_max).argmax().item()
            peak_values.append(score_func(weight_max[r], other_max[r]).item())
        else:
            r, c = divmod(weight.argmax().item(), weight.shape[-1])
            r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
            peak_values.append((weight[r, c] - weight[r2, c2]).item())
        exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
        exchange_ids.append(exchange_id)

        weight = _weight.detach().clone()
        if other is not None:
            other = _other.detach().clone()
        select_length = torch.argmin(torch.tensor(peak_values)).item()
        exchange_ids = exchange_ids[:select_length]
        peak_values = peak_values[:select_length+1]

        R_ = torch.eye(self.block_size).to(weight)
        for exchange_id in exchange_ids:
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(R_)
            R_ = torch.matmul(R_, R)
        weight = torch.matmul(weight.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        if other is not None:
            other = torch.matmul(other.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        return (weight, exchange_ids, R_) if other is None else (weight, other, exchange_ids, peak_values[select_length], R_)
    
    def calculate_std(self, weight):
        weight = weight.abs().max(dim=0).values
        groups = [weight[j * self.block_size: (j+1) * self.block_size] for j in range(weight.shape[0] // self.block_size)]
        group_means = [sum(group) / len(group) for group in groups]
        mean = sum(group_means) / len(group_means)
        variance = sum((x - mean) ** 2 for x in group_means) / len(group_means)
        return math.sqrt(variance)

    def online_duquant_cali(self, weight):
        weight = weight.detach().clone()
        T = {}
        
        self.permutation_list = None
        self.R = None
        for i in range(self.permutation_times):
            weight, _, R = self.rotation(weight)
            if self.R is None:
                self.R = R.unsqueeze(0)
            else:
                self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
            
            weight, perm = self.permutation_zigzag(weight)
            if self.permutation_list is None:
                self.permutation_list = perm.unsqueeze(0)
            else:
                self.permutation_list = torch.cat((self.permutation_list, perm.unsqueeze(0)), dim=0)

        weight, _, R = self.rotation(weight)
        if self.R is None:
            self.R = R.unsqueeze(0)
        else:
            self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
        return weight

    def init_duquant(self, x: torch.Tensor):
        if self.quant_method is None:
            return x
        if self.rotate is None:
                x_shape = x.shape   # (n_tokens, hidden_dim) / (out_features, in_features)
                hadamard = self.H.to(x)
                x = x.reshape(-1, self.block_size)
                x = x.matmul(hadamard).view(x_shape)
        elif self.quant_method == 'duquant':
            if self.rotate:
                if not self.init_duquant_params:
                    x = self.online_duquant_cali(x)
                    self.init_duquant_params = torch.tensor(1)
                else:
                    x_size = x.shape
                    x_type = x.dtype
                    if self.permutation_list is not None:
                        for i in range(len(self.permutation_list)):
                            x = x.reshape(-1, self.block_size)
                            R = self.R[i].to(x)
                            x = x.matmul(R).reshape(x_size).squeeze(0)
                            # if False:
                            if True:
                                if len(self.permutation_list.shape) == 3:
                                    perm = (self.permutation_list[i, 0].to(x.device), self.permutation_list[i, 1].to(x.device))
                                    x[:, perm[0]], x[:, perm[1]] = x[:, perm[1]], x[:, perm[0]]
                                else:
                                    perm = self.permutation_list[i].to(x.device)
                                    x = x[:, perm]
                    if len(self.R) > 0:
                        x = x.reshape(-1, self.block_size)
                        R = self.R[-1].to(x)
                        x = x.matmul(R).reshape(x_size) 
        else:
            raise NotImplementedError
        return x
            

    def forward(self, x: torch.Tensor, return_no_quant=False):
        
        # if self.layer_name:
        #     layer_id = (my_function() - 1) // 7
        #     distri_3d(x.abs().squeeze(0).T, layer_name=self.layer_name, layer_idx=layer_id, desc="before")
        
        
        if hasattr(self, 'smooth_scales'):
            x /= self.smooth_scales.to(x.device)

        # if self.n_bits >= 16 or not self.enable:
        #     return x
        # if self.metric == "fix0to1":
        #     return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        

            
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            x = self.init_duquant(x)
            
        # if self.layer_name:
        #     distri_3d(x.abs().squeeze(0).T, layer_name=self.layer_name, layer_idx=layer_id, desc="after")
        
        if self.w and self.inp_for_w != None:
            self.inp_for_w = self.init_duquant(self.inp_for_w)
            
            
            
            
        if self.a and self.w_for_a != None:
            self.w_for_a = self.init_duquant(self.w_for_a)
            
            
            # self.a = False
            # del self.w_for_a
        # print(self.rotate, flush=True)
        # print(self.R, flush=True)
        if return_no_quant:
            reduce_shape = [-1]
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax = x.amax(reduce_shape, keepdim=True)
            if self.swc:
                xmax = self.swc*xmax
                xmin = self.swc*xmin
            elif self.lwc:
                xmax = self.sigmoid(self.upbound_factor)*xmax
                xmin = self.sigmoid(self.lowbound_factor)*xmin
            if self.lac:
                xmax = self.lac*xmax
                xmin = self.lac*xmin
            return x
        
        if self.recorded_x_max is None:
            self.recorded_x_max = x.abs().reshape(-1, x.shape[-1]).max(axis=0).values
        if self.let_s is not None:
            x /= self.let_s

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        
        org_shape = x.shape
        
        x_uq = x.reshape(-1, 32)
        
        # print(x.shape, flush=True)
        
        xmin = x_uq.amin(reduce_shape, keepdim=True).to(x.device)
        xmax = x_uq.amax(reduce_shape, keepdim=True).to(x.device)
        # if self.swc:
        #     xmax = self.swc*xmax
        #     xmin = self.swc*xmin
        # elif self.lwc:
        #     xmax = self.sigmoid(self.upbound_factor.to(x.device))*xmax
        #     xmin = self.sigmoid(self.lowbound_factor.to(x.device))*xmin
        # elif self.lac:
        #     xmax = self.lac*xmax
        #     xmin = self.lac*xmin

        if self.symmetric:
            print("exist", flush=True)
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            
            # quant_grid = torch.tensor([0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
            #                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).to(abs_max.device).to(abs_max.dtype)
            # max_quant_val = max(quant_grid)
            
            # exp = torch.floor(torch.log2(abs_max)) - torch.floor(torch.log2(max_quant_val))
            # # scales = (max_val * alpha) / max_quant_val
            # self.scale = torch.pow(2, exp)
            
            # print("symmetric", flush=True)
    

            
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            round_method = "mrq_naive"
            
            
            abs_max = torch.max(xmax.abs(),xmin.abs())
            # scale = abs_max / (2**(self.n_bits-1)-1)
            
            quant_grid = torch.tensor([0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).to(abs_max.device).to(abs_max.dtype)
            
            # quant_grid = torch.tensor([0.0,  1.0,  2.0,  3.0,  4.0, 5.0,  6.0, 7.0,
            #                           -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]).to(abs_max.device).to(abs_max.dtype)
            
            max_quant_val = max(quant_grid)
            
            if self.a and self.w_for_a != None:
                search_tw = True
                if search_tw:
                    # search clipping ratio of activation in tensor-wise
                    min_clip_r = 0.9
                    max_clip_r = 1.5
                    
                    
                    opt_mse = 100000.0
                    if len(x.shape) > 2:
                        x = x.reshape(x.shape[-2], x.shape[-1])
                    
                    origin_output = torch.mm(x, (self.w_for_a.T))
                    
                    origin_input = x.reshape(-1, 32)
                    
                    origin_max = origin_input.abs().amax(dim=1, keepdim=True)
                    
                    
                    
                    while max_clip_r - min_clip_r > 0.005:
                        step = (max_clip_r - min_clip_r) / 5
                        for clip_r in np.arange(min_clip_r, max_clip_r+step, step):
                            
                            d_input = x.reshape(-1, 32)
                            d_input, _, _, _ = get_quant_act_mxfp(x=d_input, quant_grid=quant_grid, zero_point=False, round_method="rtn", x_clip_r=clip_r)
                            d_input = d_input.reshape(x.shape)
                            
                            # minimize the quantization error of outliers
                            d_output = torch.mm(d_input, self.w_for_a.T)
                            
                            
                            # tmp_mse = ((d_max - origin_max).to(torch.float32) / origin_max.to(torch.float32)).abs().mean(dim=0)
                            
                            # kl divergence
                            # div = d_output.abs() / origin_output.abs()
                            # div = torch.where(torch.isinf(div), torch.ones_like(div).to(div.device), div)
                            # div = torch.where(torch.isnan(div), torch.ones_like(div).to(div.device), div)
                            
                            # kl_diver = ((div) * d_output.abs()).to(torch.float32).mean(dim=1).sum(dim=0)
                            
                            # tmp_mse = (d_output - origin_output).abs().to(torch.float32).sum(dim=1).mean(dim=0)
                            
                            
                            
                            # tmp_mse = (d_output - origin_output).abs().to(torch.float32).amax(dim=1).mean(dim=0)
                            
                            tmp_mse = (d_output - origin_output).abs().to(torch.float32).sum(dim=0).mean(dim=0)
                            # print(kl_diver)
                            # print(tmp_mse)
                            if tmp_mse < opt_mse:
                                opt_mse = tmp_mse
                                opt_clip_r = clip_r
                                
                                
                        max_clip_r = opt_clip_r + step
                        min_clip_r = opt_clip_r - step
                        
                    self.lac = opt_clip_r
                    # self.x_clip_tw = opt_clip_r
                    # d_input = input.reshape(-1, 32)
                    # d_input, _, _, _ = get_quant_act_mxfp(x=d_input, weight=None, quant_grid=self.quant_grid, zero_point=False, round_method="up", x_clip_r=opt_clip_r)
                    # deq_input = d_input.reshape(input.shape)
                    
                    print(f"Opt clip ratio: {opt_clip_r}", flush=True)
                    
                    self.a = False
                    del self.w_for_a
            # if self.lac:
            #     print(self.lac, flush=True)
                    
                
            if self.swc:
                xmax = self.swc*xmax
                xmin = self.swc*xmin
            elif self.lwc:
                xmax = self.sigmoid(self.upbound_factor.to(x.device))*xmax
                xmin = self.sigmoid(self.lowbound_factor.to(x.device))*xmin
            elif self.lac:
                xmax = self.lac*xmax
                xmin = self.lac*xmin
            
            if self.w and self.inp_for_w != None:
                
                for i1 in range(0, x.shape[-1], 32):
                    i2 = min(i1 + 32, x.shape[-1])
                    # deq_weight[i*N : (i+1)*N, :], _, up_ratio = get_quant_weight_mxfp(weight[i*N : (i+1)*N, :], input_x=input[i*M : (i+1)*M, :], quant_grid=self.quant_grid, zero_point=False, round_method="up")
                    if self.scale is None:
                        self.scale = get_quant_weight_mxfp(x[:,i1:i2], input_x=self.inp_for_w[:,i1:i2], quant_grid=quant_grid, zero_point=False, round_method="w_search")
                    else:
                        tmp = get_quant_weight_mxfp(x[:,i1:i2], input_x=self.inp_for_w[:,i1:i2], quant_grid=quant_grid, zero_point=False, round_method="w_search")
                        self.scale = torch.cat((self.scale, tmp), dim=0)
                
                
                # self.scale = get_quant_weight_mxfp(w=x, quant_grid=quant_grid, input_x=self.inp_for_w, round_method="w_search")
                
                self.w = False
                del self.inp_for_w
                
            
            
            elif round_method == "mrq_naive":
                abs_max = torch.max(xmax.abs(),xmin.abs())
            
                exp = torch.floor(torch.log2(abs_max)) - torch.floor(torch.log2(max_quant_val))

                # scales = (max_val * alpha) / max_quant_val
                self.scale = torch.pow(2, exp)
                
            elif round_method == "normal":
                self.scale = abs_max / max_quant_val
            
            elif round_method == "nvfp":
                
                scales = abs_max / max_quant_val
                scales = scales.view(torch.short)
                hi = scales & 0xFF80
                r = scales & 0x0040
                
                scales = hi + r * 2
                
                self.scale = scales.view(torch.half)
                
                
            # scales_up = torch.pow(2, (torch.ceil(torch.log2(abs_max*0.88/max_quant_val))))
            # self.scale = scales_up
            
            zero_point = 6.0 * torch.ones_like(self.scale)
            
            
            # print("asymmetric", flush=True)
            # range = xmax - xmin
            # scale = range / (2**self.n_bits-1)
            # self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            # zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round()
        # print(self.round_zero_point, flush=True)
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def register_duquant_params(self):
        if self.rotate is not True:
            return
        permutation_list, R = self.permutation_list, self.R
        delattr(self, 'R')
        delattr(self, 'permutation_list')
        delattr(self, 'init_duquant_params')
        self.register_buffer('permutation_list', permutation_list)
        self.register_buffer('R', R)
        self.register_buffer('init_duquant_params', torch.tensor(1))

    def copy_duquant_params(self, quantizer_ref):
        if quantizer_ref.rotate is True:
            assert quantizer_ref.init_duquant_params == True
            self.R = quantizer_ref.R.clone().detach()
            try:
                self.permutation_list = quantizer_ref.permutation_list.clone().detach()
            except:
                self.permutation_list = quantizer_ref.permutation_list
            self.init_duquant_params = torch.tensor(1)
            

def get_quant_weight_mxfp(w, quant_grid, input_x=None, zero_point=True, q_group_size=-1, pos_value=None, round_method="rtn", error=None):
    '''
    return : dequantized weight, mse?
    '''
    quant_grid = quant_grid.to(w.device)
    
    # round_method = "normal"
    

    max_val = w.abs().amax(dim=1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    
    # RTN rounding
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_rtn = torch.pow(2, exp)
    
    # All round down
    scales_down = torch.pow(2, (torch.floor(torch.log2(max_val/max_quant_val))))
    
    # # All round up
    scales_up = torch.pow(2, (torch.ceil(torch.log2(max_val/max_quant_val))))
    
    
    
    zeros = 0
        
    # Normal rounding
    scales_n = max_val / max_quant_val
        
    if round_method == "rtn":
        scales = scales_rtn
    elif round_method == "up":
        scales = scales_up
    elif round_method == "down":
        scales = scales_down
    elif round_method == "nvfp":
        scales = scales_n.view(torch.short)
        hi = scales & 0xFF80
        r = scales & 0x0040
        
        scales = hi + r * 2
        
        scales = scales.view(torch.half)
    elif round_method == "w_search" and input_x != None:
        labels_up = (((w + zeros) / scales_up).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        labels_down = (((w + zeros) / scales_down).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        
        w_deq_up = quant_grid[labels_up] * scales_up - zeros
        w_deq_down = quant_grid[labels_down] * scales_down - zeros
        
        input_g = input_x.reshape(-1, 32)
        
        # deq_input_g, _, _, _ = get_quant_act_mxfp(x=input_g, weight=None, quant_grid=quant_grid, zero_point=False, round_method="up")
        
        out_origin = torch.mm(input_g, w.T)
        
        # out_up = torch.mm(input_g, w_deq_up.T)
        # out_down = torch.mm(input_g, w_deq_down.T)
        
        out_up = torch.mm(input_g, w_deq_up.T)
        out_down = torch.mm(input_g, w_deq_down.T)
        
        if error is None:
            up_mse = (out_origin - out_up).pow(2).mean(dim=0)
            down_mse = (out_origin - out_down).pow(2).mean(dim=0)
        else:
            up_mse = (out_up - out_origin + error).pow(2).mean(dim=0)
            down_mse = (out_down - out_origin + error).pow(2).mean(dim=0)
        
        # 1 for down, 0 for up
        mask = (((up_mse - down_mse) > 0).to(torch.int32)).unsqueeze(1)
        
        up_ratio = 1 - mask.sum() / mask.shape[0]
        
        scales = scales_down * mask + scales_up * (1 - mask)
    elif round_method == "normal":
        scales = scales_n
        

    labels = (((w + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    w_deq = quant_grid[labels] * scales - zeros

    
    quant_mse = (w_deq-w).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # if get_labels:
    #     return w_deq, quant_mse_sum, labels, quant_grid * scales
    # else:
    if round_method == "w_search":
        return scales
    else:
        return w_deq, quant_mse_sum, 1, scales







def distri_3d(w_data, group_size=-1, layer_idx=0, layer_name="", max_fig=1000, desc=""):
    if group_size > 0 and w_data.shape[-1] % group_size != 0:
        print(f"Input channel: {w_data.shape[-1]} is not divisible by group_size: {group_size}")
        return
    
    # Prepare data based on group_size
    w_data_group = w_data.reshape(-1, group_size) if group_size > 0 else w_data

    # Prepare 3D visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare X, Y, and Z values
    group_indices = np.arange(w_data_group.shape[0])
    element_indices = np.arange(w_data_group.shape[1])
    X, Y = np.meshgrid(element_indices, group_indices)
    Z = np.abs(w_data_group.cpu().numpy())  # Use absolute value for better visualization

    # Plot surface
    percentile_range = [10, 99.7]  # 10%到99%的分位数范围
    z_min, z_max = np.percentile(Z, percentile_range)  
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8, vmin=z_min, vmax=z_max)
    # surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8, norm=LogNorm(vmin=z_min, vmax=z_max))
    # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)  # 添加颜色条
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, extend='both')  # extend='both' 确保颜色条包含超出范围的值

    # 标注最大值
    max_idx = np.unravel_index(Z.argmax(), Z.shape)
    ax.text(X[max_idx], Y[max_idx], Z.max(), f'max: {Z.max():.2f}', color='red')
        # 添加分位数信息到图像顶部
    fig.text(
        0.5, 0.8,  # 中心对齐，位于顶部
        f"Percentile Range: {percentile_range[0]}% = {z_min:.4f}, {percentile_range[1]}% = {z_max:.4f}",
        fontsize=12,
        color='red',
        ha='center',  # 水平居中
        va='top',  # 垂直顶部对齐
        bbox=dict(boxstyle="round", edgecolor="black", facecolor="white", alpha=0.8)
    )

    # elev (Elevation): 从 z 轴方向观察的角度（俯仰角），默认值是 30°。越小视角越接近xy平面，值越大越类似俯视
    # azim (Azimuth): 绕 z 轴旋转的角度（方位角）。
    ax.view_init(elev=10, azim=20)

    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Output Dimension (Group Index)')
    ax.set_zlabel('Absolute Value of W')
    ax.set_title(f"Layer {layer_idx} - {layer_name} - 3D Distribution")

    # Save figure
    # current_time = datetime.now().strftime("_%m%d%H%M")  # 格式: 月日时分 (_01071127)

    save_path = os.path.join(os.getcwd(), 'distri_img')
    os.makedirs(save_path, exist_ok=True)
    # file_name = f"layer{layer_idx}_{layer_name}_3d_{desc}.png"
    file_name = f"layer{layer_idx}_{layer_name}_3d_{desc}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=600)
    plt.close()

    print(f"3D distribution plot saved at: {os.path.join(save_path, file_name)}")
    

def get_quant_act_mxfp(x, quant_grid, weight=None, zero_point=True, q_group_size=-1, pos_value=None, round_method="rtn", x_clip_r=1.0):
    '''
    return : dequantized weight, mse?
    '''
    
    # round_method = "normal"
    
    # print(x.shape, flush=True)
    quant_grid = quant_grid.to(x.device)
    
    max_val = x.abs().amax(dim=1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    
    # Normal rounding
    scales_n = max_val / max_quant_val
    
    # RTN rounding
    exp = torch.floor(torch.log2(max_val * x_clip_r)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_rtn = torch.pow(2, exp)
    
    # All round down
    scales_down = torch.pow(2, (torch.floor(torch.log2(max_val/max_quant_val))))
    
    # # All round up
    scales_up = torch.pow(2, (torch.ceil(torch.log2(max_val * x_clip_r/max_quant_val))))
    
    
    zeros = 0
        
    if round_method == "rtn":
        scales = scales_rtn
    elif round_method == "up":
        scales = scales_up
    elif round_method == "down":
        scales = scales_down
    elif round_method == "x_search" and weight != None:
        labels_up = (((x + zeros) / scales_up).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        labels_down = (((x + zeros) / scales_down).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        
        x_deq_up = quant_grid[labels_up] * scales_up - zeros
        x_deq_down = quant_grid[labels_down] * scales_down - zeros
        
        
        out_origin = torch.mm(x, weight.T)
        
        out_up = torch.mm(x_deq_up, weight.T)
        out_down = torch.mm(x_deq_down, weight.T)
        
        up_mse = (out_origin - out_up).pow(2).mean(dim=1)
        down_mse = (out_origin - out_down).pow(2).mean(dim=1)
        
        # 1 for down, 0 for up
        mask = (((up_mse - down_mse) > 0).to(torch.int32)).unsqueeze(1)
        
        up_ratio = 1 - mask.sum() / mask.shape[0]
        # print(mask.shape, flush=True)
        # print(scales_down.shape, flush=True)
        
        scales = scales_down * mask + scales_up * (1 - mask)
        
    elif round_method == "bit_op_rtn":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        r_scales_bi = scale_s_e.view(torch.half)
        scales = r_scales_bi
        
    elif round_method == "bit_op_down":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        
        # scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        sign_bit = (max_val_bi & 0x0200) << 1
        r_scales_bi = (scale_s_e - (0x0400 - sign_bit)).view(torch.half)
        
        scales = r_scales_bi
        
    elif round_method == "bit_op_up":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        
        # scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        sign_bit = (max_val_bi & 0x0200) << 1
        r_scales_bi = (scale_s_e + sign_bit).view(torch.half)
        
        scales = r_scales_bi
        
        # print(f"Bit op scale up : {scales}", flush=True)
        # print(f"Normal scale up : {scales_up}", flush=True)
        # print(f"Scaling factor error : {(scales_up - scales).T} \n", flush=True)
        
    elif round_method == "normal":
        scales = scales_n
        
    # print(x.shape, flush=True)
    labels = (((x + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    x_deq = quant_grid[labels] * scales - zeros
    
    
    # deal with zero channels
    
    # channel_sum = x_deq.sum(dim=0, keepdim=True)
    # zero_mask = (channel_sum == 0).to(torch.int32)
    
    # # assert (zero_mask == 1).any() == False, f"zero input channel, {zero_mask.tolist()}"
    # x_deq = x_deq + zero_mask * x 
    
    quant_mse = (x_deq - x).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # if get_labels:
    #     return w_deq, quant_mse_sum, labels, quant_grid * scales
    # else:
    if round_method == "x_search":
        return x_deq, quant_mse_sum, up_ratio, labels
    else:
        return x_deq, quant_mse_sum, 1, labels
    
    
def my_function():
    my_function.calls = getattr(my_function, 'calls', 0) + 1  # 初始化或递增
    return my_function.calls
    
# def counter():
#     count = 0  # 闭包变量（类似静态变量）

#     def increment():
#         nonlocal count
#         count += 1
#         return count

#     return increment()