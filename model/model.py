from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: 归一化的维度大小
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        # nn.Parameter: 将tensor注册为可学习参数，会自动加入optimizer
        # torch.ones(dim): 创建全1的tensor作为缩放参数
        self.g = nn.Parameter(torch.ones(dim)) # γ伽马项

    def _norm(self, x):
        """
        RMSNorm的核心计算：x / sqrt(mean(x^2) + eps)
        """
        # x.pow(2): 对x每个元素平方
        # .mean(-1, keepdim=True): 在最后一维求均值，保持维度
        # torch.rsqrt(): 计算平方根的倒数，即 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，shape为[batch, seq_len, dim]
        Returns:
            归一化后的tensor
        """
        # .float(): 转换为float32进行计算，提高数值稳定性
        # .type_as(x): 将结果转换回x的原始数据类型
        # self.g : 可学习的缩放参数
        output = self._norm(x.float()).type_as(x)
        return output * self.g

def precompute_freqs_cis(dim: int, end: int=int(32*1024), rope_base: float=1e6,rope_scaling: Optional[dict]=None):
    """
    计算频率矩阵
    Args:
        dim: 频率矩阵的维度
        end: 频率矩阵的结束索引
        rope_base: 频率矩阵的缩放因子
    Returns:
        频率矩阵
    """
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs,attn_factor = (1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
                         1.0)
    if rope_scaling is not None:
        if rope_scaling["type"] == "yarn":
            pass
            # Yarn scaling:
            # https://github.com/ml-explore/Yarn/blob/main/yarn/yarn.py#L42
            # https://arxiv.org/abs/2309.08580
            # https://github.com/ml-explore/Yarn/blob/main/yarn/yarn.py#L
