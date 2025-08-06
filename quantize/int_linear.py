import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import time

import numpy as np

import os 
import datetime


static_var = 0

class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        rotate=True,
        layer_id=None,
        layer_name=None
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,rotate=rotate)
        self.weight_quantizer.layer_name = layer_name
        self.weight_quantizer.layer_id = layer_id
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,rotate=rotate)
            self.act_quantizer.draw = True
            self.act_quantizer.layer_name = layer_name
            self.act_quantizer.layer_id = layer_id
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_duquant_params = torch.tensor(0) if weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)
        self.layer_id = layer_id
        self.layer_name = layer_name


    def forward(self, input: torch.Tensor):
        if input.device == 'cpu':
            print(input.device, flush=True)
        input = input.to(self.weight.device)
        # input = input.to(weight.dtype)
        if self.use_act_quant and not self.disable_input_quant:
            # print(input.shape)
        
            # exit()
            if self.act_quantizer.a:
                self.act_quantizer.w_for_a = nn.Parameter(self.weight)
            input = self.act_quantizer(input)
            # print(input.shape)
            
            # distri_3d(input.abs().squeeze(0).T, layer_name=self.layer_name, layer_idx=layer_id, desc="after")
            
            
            # exit()
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            if not self.init_duquant_params:
                self.weight_quantizer.copy_duquant_params(self.act_quantizer)
                self.init_duquant_params = torch.tensor(1) 
                
            if self.weight_quantizer.w:
                self.weight_quantizer.inp_for_w = input
            # print(f'{self.layer_name} weight {self.weight.dtype}')
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
            
        # if input.dtype != weight.dtype or (bias is not None and bias.dtype != weight.dtype):
        #     print(f'{self.layer_name} input {input.dtype}, weight {weight.dtype}, bias {bias.dtype}', flush=True)
        # #     # weight = weight.to(torch.half)
        #     input = input.to(weight.dtype).to(weight.device)
        #     bias = bias.to(weight.dtype)
        
        
        
        out = self.fwd_func(input.to(weight.dtype), weight, bias, **self.fwd_kwargs)
        
        
        
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def copy_quantizers_duquant_params(self, proj):
        assert proj.init_duquant_params
        self.init_duquant_params = torch.tensor(1)
        self.weight_quantizer.copy_duquant_params(proj.weight_quantizer)
        self.act_quantizer.copy_duquant_params(proj.act_quantizer)





