# RoPE 旋转位置编码:从公式到代码

> 补充笔记,衔接 Day 03(`wpe` 绝对位置嵌入)与 Day 04(注意力;末尾 "Beyond GPT-2" 表格里提到的 RoPE 即此)。
> GPT-2 用学习出来的绝对位置嵌入,RoPE 是 Llama / Qwen / Mistral / Gemma 等现代 LLM 的标准做法。

---

## 1. 回顾:GPT-2 的绝对位置嵌入及其局限

Day 03 中,位置信息在**嵌入层一次性**加入:

$$
x_i = \mathrm{wte}[t_i] + \mathrm{wpe}[i], \qquad \mathrm{wpe} \in \mathbb{R}^{1024 \times 768}
$$

三个局限:

1. **硬上限 1024**——位置 1024 及以后没有对应的行,模型物理上无法处理更长的序列
2. **学到的是绝对位置**——注意力得分 $\tilde q_i \cdot \tilde k_j$ 里没有任何机制保证它关心"相距多远",只能靠模型从加过 wpe 的向量里自己"悟"
3. **零参数 vs 有参数**——wpe 是一张约 0.79M 参数的学习表,而 RoPE 没有任何可学习参数

## 2. 目标:让 $q_i \cdot k_j$ 只依赖相对距离

注意力中**唯一**用到位置信息的地方是打分阶段。若能构造一个"按位置改造向量"的函数 $f$,满足

$$
\langle f(q, i),\; f(k, j) \rangle = g(q, k,\; j - i)
$$

——得分只依赖内容 $q, k$ 和**相对距离** $j-i$,与绝对位置无关——那么相对位置编码就"免费"注入了注意力。RoPE 的答案:**把向量按其位置旋转**。

## 3. 二维情形:旋转的魔法

先看 $d = 2$。定义旋转矩阵

$$
R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix},
\qquad R(m\theta)^{\mathsf T} = R(-m\theta), \qquad R(a)\,R(b) = R(a+b)
$$

把位置 $i$ 的查询、位置 $j$ 的键各旋转自己的角度:

$$
\tilde q_i = R(i\theta)\, q, \qquad \tilde k_j = R(j\theta)\, k
$$

于是得分:

$$
\tilde q_i \cdot \tilde k_j
= \big(R(i\theta)\,q\big)^{\mathsf T} \big(R(j\theta)\,k\big)
= q^{\mathsf T}\, R(i\theta)^{\mathsf T} R(j\theta)\; k
= q^{\mathsf T}\, R\big((j-i)\theta\big)\; k
$$

**绝对位置 $i, j$ 消失了,只剩 $j - i$。** 这就是 RoPE 的全部核心;正交性 $R(a)^{\mathsf T}R(b) = R(b-a)$ 一行推完。

复数视角(原论文 RoFormer 的写法):把二维向量看成复数,旋转 = 乘 $e^{\mathrm{i} m\theta}$:

$$
\langle q\, e^{\mathrm{i} m\theta},\; k\, e^{\mathrm{i} n\theta} \rangle
= \mathrm{Re}\!\big(q\, \overline{k}\; e^{\mathrm{i}(m-n)\theta}\big)
\quad\Longrightarrow\quad \text{只依赖 } m - n
$$

## 4. 推广到 $d$ 维:多频率配对旋转

$d$ 维没法整体转,但可以切成 $d/2$ 个独立的二维平面,各自用**不同频率**转。对 $d = 64$(一个注意力头):

$$
\theta_p = 10000^{-2p/d}, \qquad p = 0, 1, \dots, d/2 - 1
$$

把 $(x_{2p},\, x_{2p+1})$ 这对相邻维度看作一个二维向量,位置 $m$ 的旋转写成分量式:

$$
\begin{pmatrix} x'_{2p} \\[2pt] x'_{2p+1} \end{pmatrix}
=
\begin{pmatrix}
\cos m\theta_p & -\sin m\theta_p \\[2pt]
\sin m\theta_p & \cos m\theta_p
\end{pmatrix}
\begin{pmatrix} x_{2p} \\[2pt] x_{2p+1} \end{pmatrix},
\qquad p = 0, \dots, d/2-1
$$

整体等价于一个块对角旋转矩阵 $R(\Theta_m) = \mathrm{diag}\big(R(m\theta_0), \dots, R(m\theta_{d/2-1})\big)$,且同样满足

$$
\tilde q_i \cdot \tilde k_j = q^{\mathsf T}\, R\big(\Theta_{j-i}\big)\, k
$$

**为什么用多个频率?** 每一对维度是一根转速不同的"表针"。$d = 64$ 时(实测数值):

| $p$ | $\theta_p$ | 波长 $\lambda_p = 2\pi/\theta_p$ | 类比 |
|:-:|:-:|:-:|:-|
| 0 | $1.0$ | $\approx 6$ tokens | 秒针:最细粒度 |
| 8 | $10^{-1}$ | $\approx 63$ tokens | 分针 |
| 16 | $10^{-2}$ | $\approx 628$ tokens | |
| 24 | $10^{-3}$ | $\approx 6{,}283$ tokens | |
| 31 | $1.33 \times 10^{-4}$ | $\approx 47{,}117$ tokens | 时针:覆盖超长距离 |

所有表针的组合读数在约 $10000 \times 2\pi$ 范围内近似唯一地编码位置:快指针分辨相邻 token,慢指针分辨"远不远",远近信息同时在场。

## 5. RoPE 的关键性质

| 性质 | 出处 |
|---|---|
| 相对性:$\tilde q_i \cdot \tilde k_j$ 只依赖 $j-i$ | $R(i\theta)^{\mathsf T} R(j\theta) = R((j-i)\theta)$ |
| 保范数:$\lVert \tilde q \rVert = \lVert q \rVert$ | 旋转是正交变换,不放大不缩小 |
| 零参数 | $\theta_p$ 由公式算出,无任何可学习权重 |
| 逐层注入 | 在**每层** attention 内作用于 $q, k$;**不作用于 $v$**(打分才需要位置,载荷不需要) |
| 不占嵌入表 | Llama 里不存在 wpe 那张表 |

与 GPT-2 对比:

| | GPT-2 `wpe` | RoPE |
|---|---|---|
| 类型 | 学习的**绝对**位置 | 公式计算的**相对**位置 |
| 注入点 | 嵌入层一次性加到 $x$ 上 | 每层 attention 前旋转 $q, k$ |
| 参数量 | $1024 \times 768 \approx 0.79$M | 0 |
| 最大长度 | 硬上限(表行数) | 无表格上限,外推性取决于频率设计 |
| 作用于 $v$? | 是(加在 $x$ 上,后续全部携带) | 否 |

## 6. 代码:numpy 从零实现(含验证)

```python
import numpy as np

def rope_tables(max_pos, d, base=10000.0):
    """每个位置、每个频率对的角度表。返回 cos/sin,形状 (max_pos, d/2)"""
    p = np.arange(d // 2)                      # 频率编号 p = 0..d/2-1
    theta_p = base ** (-2.0 * p / d)           # θ_p = 10000^(-2p/d)
    m = np.arange(max_pos)[:, None]            # 位置 m = 0..max_pos-1
    ang = m * theta_p[None, :]                 # (max_pos, d/2)
    return np.cos(ang), np.sin(ang)

def apply_rope(x, cos, sin):
    """x: (..., d)。相邻两维 (2p, 2p+1) 配成一对做 2D 旋转(interleaved 约定)"""
    x_even = x[..., 0::2]                      # x_{2p}
    x_odd  = x[..., 1::2]                      # x_{2p+1}
    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out
```

**验证 1——相对性**(同一对 $q, k$,相对距离恒为 3,放在不同绝对位置,得分必须相等):

```python
d, MAX = 64, 256
rng = np.random.default_rng(0)
q, k = rng.standard_normal(d), rng.standard_normal(d)
cos, sin = rope_tables(MAX, d)

for i in [0, 5, 50, 200]:
    qi = apply_rope(q, cos[i], sin[i])
    kj = apply_rope(k, cos[i+3], sin[i+3])
    print(f"pos {i:3d} vs {i+3:3d} -> {qi @ kj:+.10f}")
```

实际输出——四个得分**逐位相同**:

```
pos   0 vs   3 -> -13.0496573491
pos   5 vs   8 -> -13.0496573491
pos   50 vs  53 -> -13.0496573491
pos  200 vs 203 -> -13.0496573491
```

**验证 2——对照组**:GPT-2 式绝对位置嵌入(`(q + wpe[i]) · (k + wpe[j])`)做同样实验,得分随绝对位置漂移,不满足相对性:

```
pos   0 vs   3 -> -10.123346
pos   5 vs   8 -> -10.161211
pos   50 vs  53 ->  -9.146322      ← 明显不同
pos 200 vs 203 -> -10.169596
```

**验证 3——保范数**:`np.allclose(norm(x), norm(apply_rope(x)))` 为 `True`(旋转不改变向量长度)。

> 实现约定提示:上面用的是 **interleaved**(相邻维配对,$x_{2p}, x_{2p+1}$)约定,即 Llama 原始代码的排布;HuggingFace 的 Llama 实现用 **rotate_half**(前半/后半配对)约定,两者只差一个固定的维度重排,数学上完全等价。读不同代码库时注意区分。

## 7. 工程视角(连接 Day 05–07)

1. **与 KV cache 的配合**(Day 05):旋转是在算 $q, k$ 之后、打分之前做的,所以缓存里存的是**旋转后的 $k$**;decode 时新 token 的旋转角度由 `len(cache)` 直接给出——day05 里 `decode_step` 的 `past_seq_len` 参数正是干这个的。RoPE 下"位置"不再查表,而是"你在缓存第几行"。
2. **长上下文外推**:RoPE 没有表格硬上限,但训练长度之外外推会掉精度。生产做法是调频率:位置插值(Position Interpolation,等比缩小角度)、NTK-aware scaling(调大 base)、YaRN(分频率处理)。一句话:**改 $\theta_p$ 的标度,而不是加表行**。
3. **成本**:每层每头对 $q, k$ 各一次逐元素旋转——$O(n \cdot d)$,相对矩阵乘可忽略,也常被 kernel fusion 并进 QKV 投影的 epilogue(Day 07)。

## 8. 一页总结

$$
\tilde q_i = R(\Theta_i)\, q_i, \qquad \tilde k_j = R(\Theta_j)\, k_j, \qquad
\tilde q_i \cdot \tilde k_j = q_i^{\mathsf T} R(\Theta_{j-i})\, k_j
$$

旋转矩阵按对分块、频率 $\theta_p = 10000^{-2p/d}$ 多尺度铺开;得分只剩相对距离 $j-i$;保范数、零参数、逐层作用于 $q/k$ 而不作用于 $v$。

## 参考

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*(arXiv:2104.09864)——RoPE 原始论文
- Chen et al., *Extending Context Window via Position Interpolation*(arXiv:2306.15595)
- Peng et al., *YaRN: Efficient Context Window Extension*(arXiv:2403.09646)
