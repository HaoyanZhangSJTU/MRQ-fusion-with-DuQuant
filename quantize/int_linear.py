import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params,rotate=rotate)
            self.act_quantizer.draw = True
            self.act_quantizer.layer_name = layer_name
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_duquant_params = torch.tensor(0) if weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)
        self.layer_name = layer_name


    def forward(self, input: torch.Tensor):
        if self.use_act_quant and not self.disable_input_quant:
            # print(input.shape)
            
            # # draw distribution before rotation
            # dis_before = input.abs().squeeze_(0).to("cpu").numpy()
            # x = dis_before.shape[0]
            # y = dis_before.shape[1]
            
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="3d")
            
            # x_line = np.array(np.arange(0, dis_before.shape[1]).tolist() * dis_before.shape[0])
            # y_tmp = np.arange(0, dis_before.shape[0])
            # y_line = np.repeat(y_tmp, dis_before.shape[1])
            # # print(y_line)
            
            # # X, Y = np.meshgrid(x_line, y_line)
            # X, Y = np.meshgrid(dis_before.shape[1], dis_before.shape[0])
            # # Z = np.abs(w_data_group.cpu().numpy()) 
            # z_line = dis_before
            
            # percentile_range = [10, 99.7]  # 10%到99%的分位数范围
            # z_min, z_max = np.percentile(z_line, percentile_range)  
            # surface = ax.plot_surface(X, Y, z_line, cmap='viridis', edgecolor='none', alpha=0.8, vmin=z_min, vmax=z_max)
            # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, extend='both')
            
            # # ax.plot(x_line, y_line, z_line, color='')
            
            
            # # for i in range(x // 128):
            # #     for j in range(y):
            # #         ax.bar3d(i, j, 0, 1, 1, dis_before[i * 128 : (i+1) * 128, j].mean())
                    
            # ax.set_xlabel('Channel')
            # ax.set_ylabel('Token')
            # ax.set_zlabel('Value')
            # print("Done", flush=True)
            
            # plt.savefig("/root/DuQuant/imgs/dis.png")
            # plt.show()


            
            # layer_id = (my_function() - 1) // 7
            # distri_3d(input.abs().squeeze(0).T, layer_name=self.layer_name, layer_idx=layer_id, desc="before")
            
            # exit()
            if self.act_quantizer.a:
                self.act_quantizer.w_for_a = self.weight
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
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def copy_quantizers_duquant_params(self, proj):
        assert proj.init_duquant_params
        self.init_duquant_params = torch.tensor(1)
        self.weight_quantizer.copy_duquant_params(proj.weight_quantizer)
        self.act_quantizer.copy_duquant_params(proj.act_quantizer)











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