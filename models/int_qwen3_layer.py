# models/int_qwen3_layer.py
# coding=utf-8
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  
from transformers.models.qwen3.modeling_qwen3 import (                
    Qwen3DecoderLayer as HF_Qwen3DecoderLayer,                       
    Qwen3Attention as HF_Qwen3Attention,                           
    Qwen3MLP as HF_Qwen3MLP,                                        
    Qwen3RotaryEmbedding,                                             
    apply_rotary_pos_emb,                                             
)
from transformers.models.qwen3.modeling_qwen3 import repeat_kv        # 与 Qwen3 等价
from transformers.activations import ACT2FN                           

from quantize.int_linear import QuantLinear                           
from quantize.int_matmul import QuantMatMul                           
from quantize.du_norm import DuQwen3RMSNorm                           

from collections import OrderedDict                                   
from models.transformation import *                                 



class QuantQwen3MLP(nn.Module):  
    def __init__(
        self,
        org_module: HF_Qwen3MLP,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: int,
        args=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = QuantLinear(
            org_module.gate_proj,
            args.gate_weight_quant_params,
            args.gate_act_quant_params,
            layer_name="gate_proj",
            layer_id=layer_id,                                  
        )
        self.down_proj = QuantLinear(
            org_module.down_proj,
            args.down_weight_quant_params,
            args.down_act_quant_params,
            layer_name="down_proj",
            layer_id=layer_id,                                  
        )
        self.up_proj = QuantLinear(
            org_module.up_proj,
            args.up_weight_quant_params,
            args.up_act_quant_params,
            layer_name="up_proj",
            layer_id=layer_id,                                  
        )
        self.act_fn = ACT2FN[hidden_act]

        # Same as int_llama_layer.py
        self.init_duquant_params = torch.tensor(
            0 if args.gate_weight_quant_params['quant_method'] == 'duquant' else 1
        )

    def forward(self, x):
        if not self.init_duquant_params:
            self.init_duquant_params = torch.tensor(1)
            act = self.act_fn(self.gate_proj(x))
            self.up_proj.copy_quantizers_duquant_params(self.gate_proj)
            mul = act * self.up_proj(x)
            return self.down_proj(mul)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------
#  q_norm/k_norm + RoPE + quant_matmul
# ---------------------------
class QuantQwen3Attention(nn.Module):  
    """Multi-headed attention for Qwen3 with quantization"""

    # Add layer_id and args for quantization
    def __init__(
        self,
        org_module: HF_Qwen3Attention,
        config: Qwen3Config,
        layer_id: int,
        args=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.layer_idx = layer_id  # [QWEN3-FIX] 供 Cache.update 使用

        # RoPE（Qwen3 版本在 Model 中计算，这里自备一份以支持 duquant 的逐层替换）
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # head-dim RMSNorm
        self.q_norm = DuQwen3RMSNorm(org_module.q_norm, eps=org_module.q_norm.variance_epsilon)
        self.k_norm = DuQwen3RMSNorm(org_module.k_norm, eps=org_module.k_norm.variance_epsilon)

        # 量化线性层
        self.k_proj = QuantLinear(
            org_module.k_proj, args.k_weight_quant_params, args.k_act_quant_params,
            layer_id=layer_id, layer_name="k_proj"
        )
        self.v_proj = QuantLinear(
            org_module.v_proj, args.v_weight_quant_params, args.v_act_quant_params,
            layer_id=layer_id, layer_name="v_proj"
        )
        self.q_proj = QuantLinear(
            org_module.q_proj, args.q_weight_quant_params, args.q_act_quant_params,
            layer_id=layer_id, layer_name="q_proj"
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.o_weight_quant_params, args.o_act_quant_params,
            layer_id=layer_id, layer_name="o_proj"
        )

        # 量化 MatMul
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None)

        self.sliding_window = getattr(config, "sliding_window", None)

        self.init_duquant_params = torch.tensor(
            0 if args.gate_weight_quant_params['quant_method'] == 'duquant' else 1
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,   
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,       # [QWEN3-FIX]，qwen3 要求该参数
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # [QWEN3-FIX]，qwen3 要求该参数
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 线性投影（首次前向对齐 duquant 量化器参数）
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if not self.init_duquant_params:
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # head-dim RMSNorm 在 RoPE 之前
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # RoPE：优先用上层算好的 cos/sin；否则用 position_ids 现算
        if position_embeddings is not None:  # [QWEN3-FIX]
            cos, sin = position_embeddings
        else:
            if position_ids is None:
                raise ValueError("[Qwen3] `position_ids` is required to compute rotary embeddings.")
            cos, sin = self.rotary_emb(value_states, position_ids)

        if cos.device != query_states.device or cos.dtype != query_states.dtype:
            cos = cos.to(device=query_states.device, dtype=query_states.dtype)   # [QWEN3-DEVFIX]
        if sin.device != query_states.device or sin.dtype != query_states.dtype:
            sin = sin.to(device=query_states.device, dtype=query_states.dtype)   # [QWEN3-DEVFIX]

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids.to(query_states.device)
        )

        # 量化前预处理
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)

        # —— 兼容两种 KV 缓存 ——  # [QWEN3-FIX]
        kv_seq_len = key_states.shape[-2]
        present_key_value = None

        # 情况 A：HF 的 Cache 实例（推荐路径）
        from transformers.cache_utils import Cache as HFCache  # 局部导入，避免硬依赖
        if isinstance(past_key_value, HFCache):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            kv_seq_len = key_states.shape[-2]

        # 情况 B：tuple(k, v) 旧式缓存
        elif past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            present_key_value = (key_states, value_states) if use_cache else None

        # n_kv_heads < n_heads 时重复扩展
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 注意力权重计算
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) * self.scaling  # [QWEN3-MATCH]
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            if causal_mask.device != attn_weights.device:
                causal_mask = causal_mask.to(attn_weights.device)
            attn_weights = attn_weights + causal_mask

        # softmax + dropout + 量化 PV
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)  # [QWEN3-MATCH]

        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)  # [bsz, nh, q_len, hd]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # 首次 duquant 懒拷贝标记置 1
        self.init_duquant_params = torch.tensor(1)

        return attn_output, attn_weights, present_key_value


# ---------------------------
# 量化版 DecoderLayer
# ---------------------------
class QuantQwen3DecoderLayer(nn.Module):  
    def __init__(
        self,
        config: Qwen3Config,
        ori_layer: HF_Qwen3DecoderLayer,
        layer_id: int,
        args,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantQwen3Attention(
            org_module=ori_layer.self_attn,
            config=config,
            layer_id=layer_id,
            args=args,
        )
        self.mlp = QuantQwen3MLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
            args=args,
        )
        self.input_layernorm = DuQwen3RMSNorm(ori_layer.input_layernorm, eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = DuQwen3RMSNorm(ori_layer.post_attention_layernorm, eps=ori_layer.post_attention_layernorm.variance_epsilon)
        self.device = ori_layer.self_attn.q_proj.weight.device

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,                 # [QWEN3-FIX]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # [QWEN3-FIX]
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.to(self.self_attn.q_proj.weight.device)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self-Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,                # [QWEN3-FIX]
            position_embeddings=position_embeddings,      # [QWEN3-FIX]
        )
        # hidden_states = residual + hidden_states
        hidden_states = (residual + hidden_states).to(residual.dtype) # dtype FIX；否则 lm_head 的输入 dtype 为 float32

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).half()
        hidden_states = self.mlp(hidden_states.to(self.mlp.up_proj.weight.device)).to(residual.device)
        # hidden_states = residual + hidden_states
        hidden_states = (residual + hidden_states).to(residual.dtype) # dtype FIX；否则 lm_head 的输入 dtype 为 float32


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    # —— DuQuant 相关接口 ——
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(
                self.input_layernorm,
                [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                self.qkv_smooth_scale, self.qkv_smooth_shift
            )
            smooth_ln_fcs_temporary(
                self.post_attention_layernorm,
                [self.mlp.up_proj, self.mlp.gate_proj],
                self.fc1_smooth_scale, self.fc1_smooth_shift
            )
            smooth_fc_fc_temporary(
                self.self_attn.v_proj, self.self_attn.o_proj,
                self.out_smooth_scale, self.out_smooth_shift
            )
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj, self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(
                self.input_layernorm,
                [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                self.qkv_smooth_scale, self.qkv_smooth_shift
            )
            smooth_ln_fcs_inplace(
                self.post_attention_layernorm,
                [self.mlp.up_proj, self.mlp.gate_proj],
                self.fc1_smooth_scale, self.fc1_smooth_shift
            )
            smooth_fc_fc_inplace(
                self.self_attn.v_proj, self.self_attn.o_proj,
                self.out_smooth_scale, self.out_smooth_shift
            )
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj, self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)

    def duquant_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)

    def duquant_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()

    def register_duquant_params(self):
        for name, module in self.named_modules():
            if isinstance(module, (QuantQwen3MLP, QuantQwen3Attention)):
                if hasattr(module, 'init_duquant_params'):
                    delattr(module, 'init_duquant_params')
                module.register_buffer('init_duquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params()
                module.act_quantizer.register_duquant_params()

    def load_duquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find('init_duquant_params') > -1:
                exec(f'self.{k} = v.to(device)')

    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=False))

    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                rg = False if k.find('down') > -1 else True
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=rg))

    def load_lwc_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('bound_factor') > -1:
                v = torch.nn.Parameter(v.to(device))
                exec(f'self.{k} = v.to(device)')
