# MokioMind 模型架构详解

## 📋 目录
- [1. 整体概述](#1-整体概述)
- [2. 核心配置类](#2-核心配置类-mokiomindconfig)
- [3. 基础组件](#3-基础组件)
- [4. 位置编码机制](#4-位置编码机制-rope)
- [5. 注意力机制](#5-注意力机制-attention)
- [6. 前馈网络](#6-前馈网络)
- [7. MoE 混合专家系统](#7-moe-混合专家系统)
- [8. Transformer Block](#8-transformer-block)
- [9. 主模型架构](#9-主模型架构-mokiomindmodel)
- [10. 因果语言模型头](#10-因果语言模型头)
- [11. 关键技术总结](#11-关键技术总结)

---

## 1. 整体概述

### 1.1 文件作用
`MokioModel.py` 是 MokioMind 项目的核心模型定义文件，实现了一个完整的基于 Transformer 架构的因果语言模型（Causal Language Model）。该模型支持：
- **标准 Transformer** 和 **MoE（Mixture of Experts）** 两种架构
- **RoPE（Rotary Position Embedding）** 位置编码，支持 YaRN 长文本扩展
- **GQA（Grouped Query Attention）** 分组查询注意力机制
- **Flash Attention** 加速训练
- **KV Cache** 推理优化

### 1.2 技术栈
- **框架**: PyTorch + Hugging Face Transformers
- **架构**: Decoder-only Transformer
- **激活函数**: SiLU (SwiGLU)
- **归一化**: RMSNorm
- **位置编码**: RoPE with YaRN scaling

### 1.3 代码结构
```
MokioModel.py
├── MokioMindConfig          # 模型配置类
├── RMSNorm                  # RMS 归一化层
├── precompute_freqs         # RoPE 频率预计算
├── apply_rotary_pos_emb     # RoPE 应用函数
├── repeat_kv                # KV 重复函数（GQA）
├── Attention                # 注意力机制
├── FeedForward              # 标准前馈网络（SwiGLU）
├── MoEGate                  # MoE 门控网络
├── MoEFeedForward           # MoE 前馈网络
├── MokioMindBlock           # Transformer Block
├── MokioMindModel           # 主干模型
└── MokioMindForCausalLM     # 因果语言模型
```

---

## 2. 核心配置类: `MokioMindConfig`

### 2.1 功能说明
继承自 `transformers.PretrainedConfig`，定义了模型的所有超参数。这是模型的"蓝图"，控制着模型的规模、结构和行为。

### 2.2 关键参数详解

#### 基础架构参数
```python
hidden_size: int = 512              # 隐藏层维度（d_model）
num_attention_heads: int = 8        # 注意力头数
num_hidden_layers: int = 8          # Transformer 层数
num_key_value_heads: int = 2        # GQA 的 KV 头数（< num_attention_heads 时启用 GQA）
intermediate_size: int = None       # FFN 中间层维度（默认自动计算为 hidden_size * 8/3 并向上取整到 64 的倍数）
vocab_size: int = 6400              # 词表大小
max_position_embeddings: int = 32768 # 最大序列长度
```

#### GQA（分组查询注意力）机制
```python
# GQA 是一种介于 MHA（多头注意力）和 MQA（多查询注意力）之间的折中方案
# - MHA: num_key_value_heads = num_attention_heads（每个头有自己的 K/V）
# - MQA: num_key_value_heads = 1（所有头共享一个 K/V）
# - GQA: 1 < num_key_value_heads < num_attention_heads（每组头共享一个 K/V）
# 
# 本例中：8 个 Q 头，2 个 KV 头 → 每 4 个 Q 头共享 1 个 KV 头
n_rep = num_attention_heads // num_key_value_heads = 8 // 2 = 4
```

#### RoPE 配置
```python
rope_theta: int = 1000000           # RoPE 基频率（越大，高频部分越密集）
inference_rope_scaling: bool = False # 是否启用 YaRN 长文本扩展
flash_attention: bool = True        # 是否使用 Flash Attention
```

#### MoE 配置（可选）
```python
use_moe: bool = False               # 是否启用 MoE
num_experts_per_tok: int = 2        # 每个 token 激活的专家数（top-k）
n_routed_experts: int = 4           # 路由专家总数
n_shared_experts: int = 1           # 共享专家数（所有 token 都会经过）
scoring_func: str = "softmax"       # 门控评分函数
aux_loss_alpha: float = 0.01        # 辅助损失系数（负载均衡）
norm_topk_prob: bool = True         # 是否归一化 top-k 权重
```

#### YaRN 缩放配置（条件生成）
```python
# 当 inference_rope_scaling=True 时，自动生成 YaRN 配置
self.rope_scaling = {
    "beta_fast": 32,                    # 高频边界 α
    "beta_slow": 1,                     # 低频边界 β
    "factor": 16,                       # 扩展倍数 s
    "original_max_position_embeddings": 2048,  # 原始训练长度
    "attention_factor": 1.0,            # 注意力温度补偿
    "type": "yarn",
} if self.inference_rope_scaling else None
```

---

## 3. 基础组件

### 3.1 RMSNorm（RMS 归一化）

#### 功能说明
RMSNorm 是 LayerNorm 的简化版本，去除了均值中心化，只保留方差归一化。计算公式：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \odot w$$

#### 代码实现
```
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # 计算 RMS（均方根）
        # x: [batch_size, seq_len, hidden_size] 或 [..., dim]
        # x.pow(2).mean(-1, keepdim=True): 沿最后一个维度求均值，保持维度 → [..., 1]
        # torch.rsqrt: 计算平方根的倒数，即 1/sqrt(x) → [..., 1]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 输出形状: [..., dim] (与输入相同)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size] 或 [..., dim]
        # 先转为 float32 计算以保证精度，再转回原类型
        return self.weight * self._norm(x.float()).type_as(x)
        # self.weight: [dim]
        # 输出形状: [..., dim] (与输入相同)
```

#### 逐行讲解
1. **`self.weight = nn.Parameter(torch.ones(dim))`**: 初始化可学习参数为全 1，相当于初始不做缩放
2. **`x.pow(2).mean(-1, keepdim=True)`**: 计算每个位置的均方值，`keepdim=True` 保持维度以便广播
3. **`torch.rsqrt(... + self.eps)`**: 计算 RMS 的倒数，加 `eps` 防止除零
4. **`x * ...`**: 将输入除以 RMS，完成归一化
5. **`.type_as(x)`**: 将结果转回输入的 dtype（如 float16/bfloat16），避免精度损失

#### 为什么用 RMSNorm？
- **计算更快**: 比 LayerNorm 少一次均值计算
- **效果相当**: 在 Transformer 中表现与 LayerNorm 相近
- **节省显存**: 不需要存储均值

---

## 4. 位置编码机制: RoPE

### 4.1 RoPE 原理简介
RoPE（Rotary Position Embedding）通过旋转矩阵将位置信息注入到注意力机制中，具有：
- **相对位置感知**: 注意力分数只依赖于 token 间的相对距离
- **外推性好**: 能更好地处理超出训练长度的序列
- **计算高效**: 只需对 Q/K 做简单的旋转操作

### 4.2 `precompute_freqs` 函数详解

#### 功能说明
预计算 RoPE 的频率张量（cos 和 sin），在推理时根据位置索引查表即可。

#### 完整代码解析
```
def precompute_freqs(
    dim: int,                              # 每个头的维度（hidden_size // num_heads）
    end: int = int(32 * 1024),             # 最大序列长度
    rope_base: float = 1e6,                # RoPE 基频率 θ
    rope_scaling: Optional[dict] = None,   # YaRN 缩放配置
):
    # ===== 第 1 步：计算标准 RoPE 频率 =====
    # 公式: θ_i = 1 / (base^(2i/d)), i ∈ [0, d/2)
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ..., dim-2]
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )
    # 此时 freqs 的形状为 [dim//2]，例如 dim=64 时为 [32]

    if rope_scaling is not None:
        # ===== 第 2 步：提取 YaRN 超参数 =====
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新"聚焦"
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),  # L₀
            rope_scaling.get("factor", 16),                               # s
            rope_scaling.get("beta_fast", 32.0),                          # α
            rope_scaling.get("beta_slow", 1.0),                           # β
            rope_scaling.get("attention_factor", 1.0),                    # k
        )

        # ===== 第 3 步：判断是否需要缩放 =====
        # 只有目标长度 > 原始训练长度时才应用 YaRN
        if end / orig_max > 1.0:
            # ===== 第 4 步：定义波长比例到维度索引的映射 =====
            # 波长 λ = 2π/θ，波长比例 b = λ/(2πL₀) = 1/(θL₀)
            # 反解出维度索引: i = (d * ln(L₀/(b*2π))) / (2*ln(base))
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # ===== 第 5 步：计算高低频切分点 =====
            # low: 波长比例 > α 的最高维度索引（高频区，不缩放）
            # high: 波长比例 < β 的最低维度索引（低频区，完全缩放）
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # ===== 第 6 步：计算混合因子 γ（Ramp 函数）=====
            # γ(i) = clamp((i - low) / (high - low), 0, 1)
            # - i < low 时: γ = 0（高频，不缩放）
            # - i > high 时: γ = 1（低频，完全缩放）
            # - low ≤ i ≤ high 时: γ ∈ (0, 1)（平滑过渡）
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                # ramp 形状: [dim//2]
                / max(high - low, 0.001),  # 防止除零
                0,
                1,
            )

            # ===== 第 7 步：频率融合公式 =====
            # θ'(i) = θ(i) * ((1-γ) + γ/s)
            # - γ=0 时: θ' = θ（保持不变）
            # - γ=1 时: θ' = θ/s（线性插值缩放）
            freqs = freqs * (1 - ramp + ramp / factor)
            # freqs 形状保持: [dim//2]

    # ===== 第 8 步：生成位置索引向量 t =====
    t = torch.arange(end, device=freqs.device)  # [0, 1, 2, ..., end-1], 形状: [end]

    # ===== 第 9 步：计算外积得到旋转角度 =====
    # freqs 形状: [dim//2], t 形状: [end]
    # torch.outer(t, freqs) → [end, dim//2]
    # freqs[t, i] = t * θ_i，表示位置 t 在第 i 个维度上的旋转角度
    freqs = torch.outer(t, freqs).float()
    # freqs 形状: [end, dim//2]

    # ===== 第 10 步：计算 cos 和 sin，并应用注意力补偿 =====
    # 扩展到 [dim] 维度：前半部分是 cos/sin，后半部分复制一份
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    # torch.cos(freqs): [end, dim//2]
    # torch.cat(..., dim=-1): [end, dim]
    # freqs_cos 形状: [end, dim]
    
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # freqs_sin 形状: [end, dim]

    return freqs_cos, freqs_sin  # 形状均为 [end, dim]
```

#### YaRN 算法核心思想
YaRN（Yet another RoPE for exteNded context length）解决了 RoPE 在长文本外推时的性能下降问题：

1. **高频不变**: 短波长（高频率）的维度已经能捕捉局部信息，无需缩放
2. **低频缩放**: 长波长（低频率）的维度需要缩放以适应更长的序列
3. **平滑过渡**: 在高低频之间使用线性插值，避免突变

**数学公式**:
$$\theta'_i = \theta_i \cdot \left((1-\gamma_i) + \frac{\gamma_i}{s}\right)$$

其中 $s$ 是扩展倍数，$\gamma_i$ 是 Ramp 函数。

### 4.3 `apply_rotary_pos_emb` 函数

#### 功能说明
将预计算的 RoPE 频率应用到 Q 和 K 上，实现位置编码。

#### 代码实现
```
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # x: [batch_size, seq_len, num_heads, head_dim] 或 [batch_size, num_heads, seq_len, head_dim]
        # 将向量的后半部分取负号后与前半部分交换
        # 例如: [a, b, c, d] → [-c, -d, a, b]
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
        # 输出形状: 与输入相同 [..., head_dim]

    # RoPE 公式: q' = q * cos(θ) + rotate_half(q) * sin(θ)
    # cos/sin 需要在 head_dim 维度上广播
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] 或 [batch_size, seq_len, head_dim]
    # cos.unsqueeze(unsqueeze_dim): 在指定维度插入新维度，以便广播到 Q/K 的形状
    #   如果 unsqueeze_dim=1, cos: [seq_len, head_dim] → [1, seq_len, 1, head_dim]
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    # q_embed 形状: [batch_size, seq_len, num_heads, head_dim]
    
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    # k_embed 形状: [batch_size, seq_len, num_heads, head_dim]
    
    return q_embed, k_embed
```

#### 逐行讲解
1. **`rotate_half(x)`**: 实现旋转操作的一半，将向量分为两半，后半部分取负后与前半部分拼接
2. **`cos.unsqueeze(unsqueeze_dim)`**: 在指定维度插入新维度，以便广播到 Q/K 的形状
3. **`q * cos + rotate_half(q) * sin`**: RoPE 的核心公式，等价于二维旋转矩阵乘法

**几何解释**: RoPE 将每个头维度视为多个二维平面，在每个平面上根据位置 t 旋转角度 θ·t。

### 4.4 `repeat_kv` 函数（GQA 实现）

#### 功能说明
将 KV 头重复以匹配 Q 头的数量，实现 GQA 机制。

```
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [batch_size, seq_len, num_key_value_heads, head_dim]
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x  # MHA 情况，无需重复

    # 步骤分解：
    # 1. x[:, :, :, None, :] → [bs, slen, num_kv_heads, 1, head_dim]
    # 2. .expand(...) → [bs, slen, num_kv_heads, n_rep, head_dim]
    # 3. .reshape(...) → [bs, slen, num_kv_heads * n_rep, head_dim]
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )
    # 输出形状: [batch_size, seq_len, num_key_value_heads * n_rep, head_dim]
    # 例如: [2, 10, 2, 64] → [2, 10, 8, 64] (当 n_rep=4)

```

**示例**: 如果有 2 个 KV 头，`n_rep=4`，则输出 8 个头，每个 KV 头重复 4 次。

---

## 5. 注意力机制: `Attention`

### 5.1 功能说明
实现了支持 GQA、Flash Attention 和 KV Cache 的多头注意力机制。

### 5.2 初始化
```
class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        # 确定 KV 头数（兼容旧配置）
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads      # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads   # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 重复倍数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # 投影层（无偏置，节省参数）
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # 检查是否支持 Flash Attention
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )
```

### 5.3 Forward 方法详解

```
def forward(
    self,
    x: torch.Tensor,                              # 输入 [bsz, seq_len, hidden_size]
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin), 每个形状为 [seq_len, head_dim]
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # KV Cache, 每个形状为 [bsz, past_seq_len, num_kv_heads, head_dim]
    use_cache=False,                              # 是否返回 KV Cache
    attention_mask: Optional[torch.Tensor] = None, # 注意力掩码 [bsz, seq_len]
):
    bsz, seq_len, _ = x.shape
    
    # ===== 第 1 步：线性投影 =====
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    # xq: [bsz, seq_len, num_attention_heads * head_dim]
    # xk: [bsz, seq_len, num_key_value_heads * head_dim]
    # xv: [bsz, seq_len, num_key_value_heads * head_dim]
    
    # ===== 第 2 步：重塑为多头形状 =====
    xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
    # xq: [bsz, seq_len, n_local_heads, head_dim]
    
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    # xk: [bsz, seq_len, n_local_kv_heads, head_dim]
    
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    # xv: [bsz, seq_len, n_local_kv_heads, head_dim]
    
    # ===== 第 3 步：应用 RoPE =====
    cos, sin = position_embeddings
    # cos, sin: [seq_len, head_dim] 或根据 start_pos 切片后的长度
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    # xq, xk 形状保持不变: [bsz, seq_len, n_heads, head_dim]
    
    # ===== 第 4 步：KV Cache 拼接 =====
    if past_key_value is not None:
        # past_key_value[0]: [bsz, past_seq_len, n_local_kv_heads, head_dim]
        # xk: [bsz, seq_len, n_local_kv_heads, head_dim]
        xk = torch.cat([past_key_value[0], xk], dim=1)  # 沿序列维度拼接
        # xk: [bsz, past_seq_len + seq_len, n_local_kv_heads, head_dim]
        
        xv = torch.cat([past_key_value[1], xv], dim=1)
        # xv: [bsz, past_seq_len + seq_len, n_local_kv_heads, head_dim]
        
    past_kv = (xk, xv) if use_cache else None  # 保存当前 KV 用于下次推理
    
    # ===== 第 5 步：转置并重复 KV =====
    # 转置为 [bsz, n_heads, seq_len, head_dim] 以符合 PyTorch 注意力格式
    xq, xk, xv = (
        xq.transpose(1, 2),                                    # [bsz, n_q_heads, seq_len, head_dim]
        repeat_kv(xk, self.n_rep).transpose(1, 2),             # [bsz, n_q_heads, seq_len, head_dim]
        repeat_kv(xv, self.n_rep).transpose(1, 2),             # [bsz, n_q_heads, seq_len, head_dim]
    )
    # 转置后:
    # xq: [bsz, n_local_heads, seq_len, head_dim]
    # xk: [bsz, n_local_heads, total_seq_len, head_dim] (total_seq_len = past_seq_len + seq_len)
    # xv: [bsz, n_local_heads, total_seq_len, head_dim]
    
    # ===== 第 6 步：计算注意力 =====
    if (
        self.flash
        and (seq_len > 1)                    # 训练阶段（seq_len > 1）
        and (past_key_value is None)         # 无 KV Cache
        and (attention_mask is None or torch.all(attention_mask == 1))  # 无掩码或全 1
    ):
        # 使用 Flash Attention（更高效）
        output = F.scaled_dot_product_attention(
            xq,   # [bsz, n_heads, seq_len, head_dim]
            xk,   # [bsz, n_heads, seq_len, head_dim]
            xv,   # [bsz, n_heads, seq_len, head_dim]
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  #  causal mask（下三角）
        )
        # output: [bsz, n_heads, seq_len, head_dim]
    else:
        # 手动实现注意力（推理阶段或有掩码时）
        
        # 6.1 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # xq: [bsz, n_heads, seq_len, head_dim]
        # xk.transpose(-2, -1): [bsz, n_heads, head_dim, total_seq_len]
        # scores: [bsz, n_heads, seq_len, total_seq_len]
        
        # 6.2 添加因果掩码（上三角填 -inf）
        # torch.triu 生成上三角矩阵，diagonal=1 表示主对角线上方
        scores[:, :, :, -seq_len:] += torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1,
        )
        # scores 形状保持: [bsz, n_heads, seq_len, total_seq_len]
        
        # 6.3 添加注意力掩码（padding mask）
        if attention_mask is not None:
            # attention_mask: [bsz, seq_len] → [bsz, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # extended_attention_mask: [bsz, 1, 1, seq_len]
            
            # 将 0 变为 -1e9，1 保持 0
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            # extended_attention_mask: [bsz, 1, 1, seq_len]
            
            scores = scores + extended_attention_mask
            # scores: [bsz, n_heads, seq_len, total_seq_len]
        
        # 6.4 Softmax + Dropout
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # scores: [bsz, n_heads, seq_len, total_seq_len]
        
        scores = self.attn_dropout(scores)
        # scores: [bsz, n_heads, seq_len, total_seq_len]
        
        # 6.5 加权求和: Attention @ V
        output = scores @ xv
        # scores: [bsz, n_heads, seq_len, total_seq_len]
        # xv: [bsz, n_heads, total_seq_len, head_dim]
        # output: [bsz, n_heads, seq_len, head_dim]
    
    # ===== 第 7 步：输出投影 =====
    output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # 合并多头
    # output.transpose(1, 2): [bsz, seq_len, n_heads, head_dim]
    # output.reshape: [bsz, seq_len, n_heads * head_dim] = [bsz, seq_len, hidden_size]
    
    output = self.resid_dropout(self.o_proj(output))
    # o_proj(output): [bsz, seq_len, hidden_size]
    # 输出形状: [bsz, seq_len, hidden_size]
    
    return output, past_kv
```

### 5.4 关键技术点

#### Flash Attention vs 手动实现
| 特性 | Flash Attention | 手动实现 |
|------|----------------|---------|
| 速度 | 快（IO 优化） | 慢 |
| 显存 | 省（不存储注意力矩阵） | 费 |
| 适用场景 | 训练阶段 | 推理阶段/有掩码 |
| 因果掩码 | `is_causal=True` | 手动添加上三角掩码 |

#### KV Cache 机制
- **训练时**: `past_key_value=None`，不使用缓存
- **推理时**: 逐步拼接历史 KV，避免重复计算
- **优势**: 将推理复杂度从 O(n²) 降至 O(n)

#### 因果掩码（Causal Mask）
确保每个位置只能看到之前的位置，防止信息泄露：
```
位置:  0    1    2    3
0     [✓   ✗   ✗   ✗]
1     [✓   ✓   ✗   ✗]
2     [✓   ✓   ✓   ✗]
3     [✓   ✓   ✓   ✓]
```

---

## 6. 前馈网络

### 6.1 SwiGLU 前馈网络: `FeedForward`

#### 功能说明
实现了 SwiGLU（Swish-Gated Linear Unit）激活的前馈网络，比传统 ReLU/GLU 效果更好。

#### 代码实现
```
class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        
        # 自动计算 intermediate_size
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到 64 的倍数（硬件优化）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # SwiGLU 的三个线性层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 默认为 "silu"

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        # SwiGLU 公式: Down(SiLU(Gate(x)) * Up(x))
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # gate_proj(x): [batch_size, seq_len, intermediate_size]
        # act_fn(...): [batch_size, seq_len, intermediate_size]
        # up_proj(x): [batch_size, seq_len, intermediate_size]
        # gated: [batch_size, seq_len, intermediate_size]
        
        return self.dropout(self.down_proj(gated))
        # down_proj(gated): [batch_size, seq_len, hidden_size]
        # 输出形状: [batch_size, seq_len, hidden_size]
```

#### SwiGLU 公式
$$\text{SwiGLU}(x) = \text{Down}\left(\text{SiLU}(\text{Gate}(x)) \odot \text{Up}(x)\right)$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（Sigmoid Linear Unit）

#### 为什么用 SwiGLU？
1. **表达能力更强**: 门控机制允许动态调整信息流
2. **训练更稳定**: 梯度流动更好
3. **业界标准**: LLaMA、PaLM 等主流模型均采用

#### intermediate_size 计算
```python
# 默认值：hidden_size * 8/3 ≈ 2.67 * hidden_size
# 例如 hidden_size=512 → intermediate_size ≈ 1365
# 向上取整到 64 的倍数：1365 → 1408
```

---

## 7. MoE 混合专家系统

### 7.1 MoE 架构概述
MoE（Mixture of Experts）通过稀疏激活机制，在增加模型容量的同时保持计算成本可控。

**核心思想**:
- 有多个"专家"（FFN 网络）
- 每个 token 只激活 top-k 个专家
- 通过门控网络决定激活哪些专家

### 7.2 MoE 门控网络: `MoEGate`

#### 功能说明
计算每个 token 对各专家的评分，选择 top-k 个专家并计算权重。

#### 代码实现
```python
class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok       # 每个 token 激活的专家数
        self.n_routed_experts = config.n_routed_experts  # 专家总数
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha            # 辅助损失系数
        self.seq_aux = config.seq_aux                 # 是否按序列计算辅助损失
        self.norm_topk_prob = config.norm_topk_prob   # 是否归一化权重
        self.gating_dim = config.hidden_size
        
        # 门控权重矩阵 [n_experts, hidden_size]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Kaiming 初始化

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # [bsz*seq_len, hidden]
        # hidden_states: [bsz * seq_len, hidden_size]
        
        # 计算每个 token 对每个专家的评分
        logits = F.linear(hidden_states, self.weight, None)  # [bsz*seq_len, n_experts]
        # self.weight: [n_routed_experts, hidden_size]
        # logits: [bsz * seq_len, n_routed_experts]
        
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)  # 归一化为概率分布
            # scores: [bsz * seq_len, n_routed_experts]
        else:
            raise NotImplementedError(...)
        
        # 选择 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # topk_weight: [bsz * seq_len, top_k]
        # topk_idx: [bsz * seq_len, top_k]
        
        # 归一化 top-k 权重（使它们的和为 1）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # denominator: [bsz * seq_len, 1]
            topk_weight = topk_weight / denominator
            # topk_weight: [bsz * seq_len, top_k]
        
        # 计算辅助损失（负载均衡）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # topk_idx_for_aux_loss: [bsz, seq_len * top_k]
            
            if self.seq_aux:
                # 按序列计算辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # scores_for_seq_aux: [bsz, seq_len, n_routed_experts]
                
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # ce: [bsz, n_routed_experts]
                
                # 统计每个专家被选中的次数
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,  # [bsz, seq_len * top_k]
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                    # ones: [bsz, seq_len * top_k]
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                # ce: [bsz, n_routed_experts]
                
                # 辅助损失 = CE * 平均评分
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                # scores_for_seq_aux.mean(dim=1): [bsz, n_routed_experts]
                # ce * ...: [bsz, n_routed_experts]
                # .sum(dim=1): [bsz]
                # .mean(): scalar
                # aux_loss: scalar
            else:
                # 按 batch 计算辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # topk_idx_for_aux_loss.view(-1): [bsz * seq_len * top_k]
                # mask_ce: [bsz * seq_len * top_k, n_routed_experts]
                
                ce = mask_ce.float().mean(0)
                # ce: [n_routed_experts]
                
                Pi = scores_for_aux.mean(0)
                # Pi: [n_routed_experts]
                
                fi = ce * self.n_routed_experts
                # fi: [n_routed_experts]
                
                aux_loss = (Pi * fi).sum() * self.alpha
                # aux_loss: scalar
        else:
            aux_loss = scores.new_zeros(1).squeeze()
            # aux_loss: scalar
        
        return topk_idx, topk_weight, aux_loss
        # topk_idx: [bsz * seq_len, top_k]
        # topk_weight: [bsz * seq_len, top_k]
        # aux_loss: scalar
```

#### 辅助损失（Auxiliary Loss）
**目的**: 防止所有 token 都涌向少数几个专家，实现负载均衡。

**公式**:
$$\mathcal{L}_{aux} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中：
- $f_i$: 专家 $i$ 被选中的频率
- $P_i$: 专家 $i$ 的平均评分
- $\alpha$: 平衡系数（通常 0.01）

### 7.3 MoE 前馈网络: `MoEFeedForward`

#### 功能说明
整合门控网络和专家层，实现稀疏激活的 MoE 机制。

#### 训练阶段 Forward
```
def forward(self, x):
    # x: [batch_size, seq_len, hidden_size]
    identity = x
    orig_shape = x.shape
    bsz, seq_len, h = orig_shape
    
    # 第 1 步：门控选择专家
    topk_idx, topk_weight, aux_loss = self.gate(x)
    # topk_idx: [bsz * seq_len, top_k]
    # topk_weight: [bsz * seq_len, top_k]
    
    # 第 2 步：展平以便处理
    x = x.view(-1, x.shape[-1])  # [bsz*seq_len, hidden]
    # x: [bsz * seq_len, hidden_size]
    
    flat_topk_idx = topk_idx.view(-1)  # [bsz*seq_len*top_k]
    # flat_topk_idx: [bsz * seq_len * top_k]
    
    if self.training:
        # ===== 训练阶段：使用重复策略 =====
        
        # 2.1 重复输入 token（每个 token 重复 top_k 次）
        x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
        # 例如：[t1, t2, t3] → [t1, t1, t2, t2, t3, t3]（top_k=2）
        # x: [bsz * seq_len * top_k, hidden_size]
        
        # 2.2 初始化输出张量
        y = torch.empty_like(x, dtype=x.dtype)
        # y: [bsz * seq_len * top_k, hidden_size]
        
        # 2.3 遍历所有专家，并行处理
        for i, expert in enumerate(self.experts):
            # 找到分配给专家 i 的 token
            mask = flat_topk_idx == i
            # mask: [bsz * seq_len * top_k], bool
            
            expert_out = expert(x[mask])
            # x[mask]: [num_tokens_for_expert_i, hidden_size]
            # expert_out: [num_tokens_for_expert_i, hidden_size]
            
            # 处理空批次（防止梯度断裂）
            if expert_out.shape[0] > 0:
                y[mask] = expert_out.to(y.dtype)
            else:
                y[mask] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
        
        # 2.4 加权求和
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        # y.view(*topk_weight.shape, -1): [bsz * seq_len, top_k, hidden_size]
        # topk_weight.unsqueeze(-1): [bsz * seq_len, top_k, 1]
        # 相乘后: [bsz * seq_len, top_k, hidden_size]
        # .sum(dim=1): [bsz * seq_len, hidden_size]
        
        y = y.view(*orig_shape)
        # y: [bsz, seq_len, hidden_size]
    
    else:
        # ===== 推理阶段：使用缓存优化 =====
        y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # moe_infer 输出: [bsz * seq_len, hidden_size]
        # y: [bsz, seq_len, hidden_size]
    
    # 第 3 步：添加共享专家输出
    if self.config.n_shared_experts > 0:
        for expert in self.shared_experts:
            y = y + expert(identity)
            # expert(identity): [bsz, seq_len, hidden_size]
            # y: [bsz, seq_len, hidden_size]
    
    self.aux_loss = aux_loss
    return y
    # 输出形状: [bsz, seq_len, hidden_size]
```

#### 推理阶段: `moe_infer` 方法
```
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    # x: [bsz * seq_len, hidden_size]
    # flat_expert_indices: [bsz * seq_len * top_k]
    # flat_expert_weights: [bsz * seq_len * top_k, 1]
    
    expert_cache = torch.zeros_like(x)
    # expert_cache: [bsz * seq_len, hidden_size]
    
    # 对专家索引排序，使相同专家的 token 连续
    idxs = flat_expert_indices.argsort()
    # idxs: [bsz * seq_len * top_k]
    
    # 统计每个专家的 token 数量（累积和）
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    # tokens_per_expert: [n_routed_experts], 一维数组
    
    # 计算每个 token 对应的原始索引
    token_idxs = idxs // self.config.num_experts_per_tok
    # token_idxs: [bsz * seq_len * top_k]
    
    # 逐个专家处理
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        if start_idx == end_idx:
            continue  # 跳过空专家
        
        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]
        # exp_token_idx: [num_tokens_for_expert_i]
        
        expert_tokens = x[exp_token_idx]
        # expert_tokens: [num_tokens_for_expert_i, hidden_size]
        
        # 批量处理当前专家的所有 token
        expert_out = expert(expert_tokens).to(expert_cache.dtype)
        # expert_out: [num_tokens_for_expert_i, hidden_size]
        
        # 加权
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        # flat_expert_weights[idxs[start_idx:end_idx]]: [num_tokens_for_expert_i, 1]
        # expert_out: [num_tokens_for_expert_i, hidden_size]
        
        # 散点加回到对应位置
        expert_cache.scatter_add_(
            0, 
            exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
            # exp_token_idx.view(-1, 1): [num_tokens_for_expert_i, 1]
            # .repeat(1, x.shape[-1]): [num_tokens_for_expert_i, hidden_size]
            expert_out
            # expert_out: [num_tokens_for_expert_i, hidden_size]
        )
        # expert_cache 保持形状: [bsz * seq_len, hidden_size]
    
    return expert_cache
    # 输出形状: [bsz * seq_len, hidden_size]
```

#### 训练 vs 推理的差异
| 特性 | 训练阶段 | 推理阶段 |
|------|---------|---------|
| 策略 | 重复输入（repeat_interleave） | 排序+分批（sorting + batching） |
| 优点 | 实现简单，梯度流畅 | 效率高，减少冗余计算 |
| 缺点 | 有冗余计算 | 实现复杂 |

#### MoE 的优势
1. **稀疏激活**: 每次只激活部分专家，计算成本低
2. **模型容量大**: 可以增加专家数量而不显著增加计算量
3. **专业化**: 不同专家可以学习不同的知识

---

## 8. Transformer Block: `MokioMindBlock`

### 8.1 功能说明
标准的 Pre-Norm Transformer Block，包含自注意力和前馈网络两个子层。

### 8.2 代码实现
```
class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.self_attention = Attention(config)
        self.layer_id = layer_id
        
        # Pre-Norm 架构：先归一化，再进入子层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 根据配置选择标准 FFN 或 MoE
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin), 每个形状为 [seq_len, head_dim]
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # KV Cache, 每个形状为 [bsz, past_seq_len, num_kv_heads, head_dim]
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, seq_len]
    ):
        res = hidden_states  # 残差连接
        # hidden_states: [bsz, seq_len, hidden_size]
        
        # ===== 第 1 个子层：自注意力 =====
        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # Pre-Norm
            # input_layernorm 输出: [bsz, seq_len, hidden_size]
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        # hidden_states: [bsz, seq_len, hidden_size]
        # present_key_value: ([bsz, total_seq_len, num_kv_heads, head_dim], [bsz, total_seq_len, num_kv_heads, head_dim]) 或 None
        
        hidden_states = res + hidden_states  # 残差连接
        # hidden_states: [bsz, seq_len, hidden_size]
        
        # ===== 第 2 个子层：前馈网络 =====
        res = hidden_states
        # res: [bsz, seq_len, hidden_size]
        
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)  # Pre-Norm
            # post_attention_layernorm 输出: [bsz, seq_len, hidden_size]
            # mlp 输出: [bsz, seq_len, hidden_size]
        )
        # hidden_states: [bsz, seq_len, hidden_size]
        
        return hidden_states, present_key_value
```

### 8.3 Pre-Norm vs Post-Norm
| 架构 | 公式 | 特点 |
|------|------|------|
| Pre-Norm | $x + \text{Sublayer}(\text{Norm}(x))$ | 训练更稳定，梯度流动好 |
| Post-Norm | $\text{Norm}(x + \text{Sublayer}(x))$ | 表达能力强，但训练困难 |

**本模型采用 Pre-Norm**，这是现代大模型的标准做法。

---

## 9. 主模型架构: `MokioMindModel`

### 9.1 功能说明
堆叠多个 Transformer Block，构成模型的主干网络。

### 9.2 初始化
```
class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # Token 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Block 列表
        self.layers = nn.ModuleList(
            [MokioMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算 RoPE 频率
        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

### 9.3 Forward 方法
```
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,  # [bsz, seq_len]
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # List of KV caches for each layer
    use_cache: bool = False,
    **kwargs,
):
    batch_size, seq_length = input_ids.shape
    # input_ids: [bsz, seq_len]
    
    # 兼容性处理（某些情况下 past_key_values 可能是对象）
    if hasattr(past_key_values, "layers"):
        past_key_values = None
    
    past_key_values = past_key_values or [None] * len(self.layers)
    # past_key_values: List of length num_hidden_layers, each element is None or (key_cache, value_cache)
    
    # 计算起始位置（用于 KV Cache）
    start_pos = (
        past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
    )
    # start_pos: int, 表示已有历史序列长度
    
    # ===== 第 1 步：Token 嵌入 + Dropout =====
    hidden_states = self.dropout(self.embed_tokens(input_ids))
    # embed_tokens(input_ids): [bsz, seq_len, hidden_size]
    # hidden_states: [bsz, seq_len, hidden_size]
    
    # ===== 第 2 步：切片 RoPE 频率 =====
    position_embeddings = (
        self.freqs_cos[start_pos : start_pos + seq_length],
        # freqs_cos: [max_position_embeddings, head_dim]
        # 切片后: [seq_length, head_dim]
        
        self.freqs_sin[start_pos : start_pos + seq_length],
        # freqs_sin: [max_position_embeddings, head_dim]
        # 切片后: [seq_length, head_dim]
    )
    # position_embeddings: ([seq_length, head_dim], [seq_length, head_dim])
    
    # ===== 第 3 步：逐层前向传播 =====
    presents = []
    for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
        hidden_states, present = layer(
            hidden_states,  # [bsz, seq_len, hidden_size]
            position_embeddings,  # ([seq_length, head_dim], [seq_length, head_dim])
            past_key_value=past_key_value,  # None or (key_cache, value_cache)
            use_cache=use_cache,
            attention_mask=attention_mask,  # [bsz, seq_len]
        )
        # hidden_states: [bsz, seq_len, hidden_size]
        # present: None or (key_cache, value_cache), 每个形状为 [bsz, total_seq_len, num_kv_heads, head_dim]
        
        presents.append(present)
    # presents: List of length num_hidden_layers
    
    # ===== 第 4 步：最终归一化 =====
    hidden_states = self.norm(hidden_states)
    # hidden_states: [bsz, seq_len, hidden_size]
    
    # ===== 第 5 步：收集 MoE 辅助损失 =====
    aux_loss = sum(
        [
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MoEFeedForward)
        ],
        hidden_states.new_zeros(1).squeeze(),
    )
    # aux_loss: scalar
    
    return hidden_states, presents, aux_loss
    # hidden_states: [bsz, seq_len, hidden_size]
    # presents: List of length num_hidden_layers
    # aux_loss: scalar
```

### 9.4 关键点说明

#### KV Cache 管理
- `past_key_values`: 列表，每个元素是一层的 `(key_cache, value_cache)`
- `start_pos`: 根据已有缓存长度确定 RoPE 的起始位置
- `presents`: 收集所有层的最新 KV，用于下次推理

#### RoPE 频率切片
```
# 假设 max_position_embeddings=32768，当前 seq_length=10，start_pos=100
# 则使用 freqs_cos[100:110] 和 freqs_sin[100:110]
# 这样位置编码是连续的：100, 101, ..., 109
```

#### MoE 辅助损失聚合
```
# 只从 MoE 层收集辅助损失
aux_loss = sum([layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MoEFeedForward)])
# 如果没有 MoE 层，返回零张量
```

---

## 10. 因果语言模型头: `MokioMindForCausalLM`

### 10.1 功能说明
在主干模型基础上添加语言模型头，支持因果语言建模任务（下一个 token 预测）。

### 10.2 初始化
```
class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        
        # 语言模型头（线性投影到词表）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重共享：嵌入层和 LM 头共享权重（节省参数，提升效果）
        self.model.embed_tokens.weight = self.lm_head.weight
```

### 10.3 Forward 方法
```
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,  # [bsz, seq_len]
    attention_mask: Optional[torch.Tensor] = None,  # [bsz, seq_len]
    labels: Optional[torch.Tensor] = None,  # 用于计算损失, [bsz, seq_len]
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # KV Cache list
    use_cache: bool = False,
    logits_to_keep: Union[int, torch.Tensor] = 0,  # 只保留最后 N 个位置的 logits
    **args,
):
    # ===== 第 1 步：主干模型前向传播 =====
    hidden_states, past_key_values, aux_loss = self.model(
        input_ids=input_ids,  # [bsz, seq_len]
        attention_mask=attention_mask,  # [bsz, seq_len]
        past_key_values=past_key_values,
        use_cache=use_cache,
        **args,
    )
    # hidden_states: [bsz, seq_len, hidden_size]
    # past_key_values: List of length num_hidden_layers
    # aux_loss: scalar
    
    # ===== 第 2 步：切片 logits（优化显存）=====
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    # hidden_states[:, slice_indices, :]: [bsz, kept_seq_len, hidden_size]
    # lm_head: Linear(hidden_size, vocab_size, bias=False)
    # logits: [bsz, kept_seq_len, vocab_size]
    
    # ===== 第 3 步：计算交叉熵损失 =====
    loss = None
    if labels is not None:
        # labels: [bsz, seq_len]
        
        # 移位：预测 t+1 位置的 token
        shift_logits = logits[..., :-1, :].contiguous()
        # logits: [bsz, kept_seq_len, vocab_size]
        # shift_logits: [bsz, kept_seq_len-1, vocab_size]
        
        shift_labels = labels[..., 1:].contiguous()
        # labels: [bsz, seq_len]
        # shift_labels: [bsz, seq_len-1]
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            # shift_logits.view(-1, vocab_size): [bsz * (kept_seq_len-1), vocab_size]
            
            shift_labels.view(-1),
            # shift_labels.view(-1): [bsz * (seq_len-1)]
            
            ignore_index=-100,  # 忽略 padding
        )
        # loss: scalar
    
    # ===== 第 4 步：构造输出 =====
    output = CausalLMOutputWithPast(
        loss=loss,  # scalar or None
        logits=logits,  # [bsz, kept_seq_len, vocab_size]
        past_key_values=past_key_values,  # List of length num_hidden_layers
        hidden_states=hidden_states,  # [bsz, seq_len, hidden_size]
    )
    output.aux_loss = aux_loss  # 附加 MoE 辅助损失, scalar
    
    return output

```

### 10.4 关键技术点

#### 权重共享
```
self.model.embed_tokens.weight = self.lm_head.weight
```
- **优点**: 减少参数量，正则化效果
- **原理**: 嵌入和投影互为逆操作，共享权重有助于学习更好的表示

#### Logits 切片优化
```
logits_to_keep: 只保留最后 N 个位置的 logits
# 应用场景：RLHF 训练中只需要最后一个 token 的 logits
```

#### 损失计算
```
# 输入:  [t0, t1, t2, t3]
# 标签:  [t1, t2, t3, t4]
# 模型预测 t0→t1, t1→t2, t2→t3, t3→t4
shift_logits = logits[:, :-1, :]   # 去掉最后一个位置
shift_labels = labels[:, 1:]       # 去掉第一个位置
```

---

## 11. 关键技术总结

### 11.1 架构亮点

| 技术 | 作用 | 实现位置 |
|------|------|---------|
| **GQA** | 平衡 MHA 的效果和 MQA 的速度 | `repeat_kv` + `Attention` |
| **RoPE + YaRN** | 支持长文本外推 | `precompute_freqs` |
| **SwiGLU** | 增强非线性表达能力 | `FeedForward` |
| **MoE** | 稀疏激活，扩大模型容量 | `MoEFeedForward` |
| **Pre-Norm** | 稳定训练，改善梯度流动 | `MokioMindBlock` |
| **KV Cache** | 加速推理 | `Attention.forward` |
| **Flash Attention** | 加速训练，节省显存 | `Attention.forward` |
| **权重共享** | 减少参数，正则化 | `MokioMindForCausalLM.__init__` |

### 11.2 训练 vs 推理差异

| 阶段 | 注意力实现 | MoE 实现 | KV Cache |
|------|-----------|---------|----------|
| **训练** | Flash Attention | repeat_interleave | 不使用 |
| **推理** | 手动实现 | sorting + batching | 使用 |

### 11.3 性能优化建议

1. **启用 Flash Attention**: 训练时速度提升 2-3 倍
2. **使用 BF16/FP16**: 减少显存占用，加速计算
3. **梯度累积**: 模拟更大的 batch size
4. **ZeRO 优化**: 分布式训练时减少显存
5. **MoE 负载均衡**: 调整 `aux_loss_alpha` 防止专家退化

### 11.4 扩展性设计

- **轻松切换 MoE**: 设置 `use_moe=True` 即可
- **调整模型规模**: 修改 `hidden_size`、`num_layers`、`num_heads`
- **长文本支持**: 启用 `inference_rope_scaling=True`
- **自定义激活函数**: 修改 `hidden_act` 参数

### 11.5 最佳实践

1. **初始化**: 使用 Kaiming 初始化保证梯度稳定
2. **Dropout**: 小模型用 0.1，大模型用 0.0-0.05
3. **学习率调度**: 使用 Cosine 退火 + Warmup
4. **监控辅助损失**: MoE 训练中关注 `aux_loss` 是否收敛
5. **验证 RoPE 外推**: 测试超出训练长度的序列性能

---

## 附录：常见问题

### Q1: 为什么 RoPE 要预计算而不是实时计算？
**A**: 预计算可以避免在前向传播中重复计算三角函数，显著提升速度。且 RoPE 频率只依赖于位置索引，与输入无关。

### Q2: GQA 如何选择 KV 头数？
**A**: 经验法则：
- 小模型（<1B）: 使用 MHA（`num_key_value_heads = num_attention_heads`）
- 中等模型（1B-10B）: GQA，`n_rep = 4-8`
- 大模型（>10B）: GQA，`n_rep = 8-16` 或 MQA

### Q3: MoE 的专家数如何确定？
**A**: 
- 专家数越多，模型容量越大，但路由开销也越大
- 推荐：`n_routed_experts = 8-64`，`num_experts_per_tok = 2-4`
- 总激活参数量 ≈ `n_routed_experts * FFN_params * (num_experts_per_tok / n_routed_experts)`

### Q4: YaRN 的超参数如何调优？
**A**: 
- `factor`: 目标长度 / 原始长度
- `beta_fast`: 32（高频边界，通常不变）
- `beta_slow`: 1（低频边界，通常不变）
- `attention_factor`: 1.0（大多数情况无需调整）

---

**文档版本**: v1.0  
**最后更新**: 2026-04-06  
**作者**: MokioMind 开发团队
