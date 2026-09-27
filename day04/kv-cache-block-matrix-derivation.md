# KV Cache 逐 token 显式推导:新来一个 token,哪些不用算、要读哪些向量

> 补充笔记,配合 `attention.ipynb` Step 8(二次方开销)与 Day 05(KV Cache)阅读。
> 全文只用 token 级向量 $q_i, k_i, v_i$ 逐元素写出注意力矩阵,所有结论直接从下标范围读出来。

---

## 1. 记号:每个 token 对应三个向量

token $t_i$ 流到某一层、某一个注意力头时($d = 64$,GPT-2 每层 12 个头各自做一遍同样的事):

$$
q_i = \mathrm{LN}(x_i)\,W_Q, \qquad
k_i = \mathrm{LN}(x_i)\,W_K, \qquad
v_i = \mathrm{LN}(x_i)\,W_V, \qquad q_i,k_i,v_i \in \mathbb{R}^{d}
$$

其中 $x_i \in \mathbb{R}^{768}$ 是 token $i$ 的隐藏状态(第 0 层即 $x_i = \mathrm{wte}[t_i] + \mathrm{wpe}[i-1]$);$W_Q, W_K, W_V$ 是 `c_attn`(768×2304)的三个 768×768 切片,训练后固定。

## 2. 得分矩阵:逐元素写出

$m$ 个 token 的得分矩阵 $S \in \mathbb{R}^{m \times m}$,第 $i$ 行第 $j$ 列就是两个 $d$ 维向量的点积:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
$$

$$
S \;=\; \frac{1}{\sqrt{d}}
\begin{array}{c|ccccc}
  & k_1 & k_2 & k_3 & \cdots & k_m \\
\hline
q_1 & q_1\!\cdot\!k_1 & q_1\!\cdot\!k_2 & q_1\!\cdot\!k_3 & \cdots & q_1\!\cdot\!k_m \\
q_2 & q_2\!\cdot\!k_1 & q_2\!\cdot\!k_2 & q_2\!\cdot\!k_3 & \cdots & q_2\!\cdot\!k_m \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
q_m & q_m\!\cdot\!k_1 & q_m\!\cdot\!k_2 & q_m\!\cdot\!k_3 & \cdots & q_m\!\cdot\!k_m
\end{array}
$$

## 3. 因果掩码:凡 $j > i$ 的元素置 $-\infty$

token $i$ 只许看过去($j \le i$),不许看未来($j > i$),mask 后是**下三角矩阵**:

$$
S_{\text{mask}} \;=\; \frac{1}{\sqrt{d}}
\begin{array}{c|ccccc}
  & k_1 & k_2 & k_3 & \cdots & k_m \\
\hline
q_1 & q_1\!\cdot\!k_1 & -\infty & -\infty & \cdots & -\infty \\
q_2 & q_2\!\cdot\!k_1 & q_2\!\cdot\!k_2 & -\infty & \cdots & -\infty \\
\vdots & \vdots & \vdots & \ddots & \ddots & \vdots \\
q_m & q_m\!\cdot\!k_1 & q_m\!\cdot\!k_2 & q_m\!\cdot\!k_3 & \cdots & q_m\!\cdot\!k_m
\end{array}
$$

右上角所有 $q_i \cdot k_j\;(j > i)$ 永远不会用到。

## 4. softmax:逐行独立

$$
a_{ij} = \frac{e^{s_{ij}}}{\sum_{l=1}^{i} e^{s_{il}}} \quad (j \le i), \qquad a_{ij} = 0 \quad (j > i)
$$

注意 $a_{ij}$ 只依赖 $q_i$ 自己和 $k_1, \dots, k_i$ 这 $i$ 个向量——分子分母都不含下标大于 $i$ 的任何东西。

## 5. 输出向量:逐 token 展开

$$
o_1 = a_{11}\,v_1
$$
$$
o_2 = a_{21}\,v_1 + a_{22}\,v_2
$$
$$
o_3 = a_{31}\,v_1 + a_{32}\,v_2 + a_{33}\,v_3
$$
$$
\boxed{\;o_i = \sum_{j=1}^{i} a_{ij}\,v_j\;}
$$

注意求和上限是 $i$ 不是 $m$。逐行读依赖:

| 要求出 | 用到的向量 |
|---|---|
| $o_1$ | $q_1,\; k_1,\; v_1$ |
| $o_2$ | $q_2,\; k_1, k_2,\; v_1, v_2$ |
| $o_i$ | $q_i,\; k_1, \dots, k_i,\; v_1, \dots, v_i$ |

第 $i$ 行永远只用前 $i$ 个 token 的 $k$ 和 $v$,且只用自己的 $q$。

## 6. 新 token $t_{n+1}$ 到来:逐元素看哪些要算

### 6.1 小例子:前面 3 个 token,$t_4$ 来了

$$
\begin{array}{c|cccc}
  & k_1 & k_2 & k_3 & k_4 \\
\hline
q_1 & \cdot & -\infty & -\infty & -\infty \\
q_2 & \cdot & \cdot & -\infty & -\infty \\
q_3 & \cdot & \cdot & \cdot & -\infty \\
q_4 & \star & \star & \star & \star
\end{array}
\qquad
\begin{aligned}
\cdot \;\; &\text{原样保留,不用重算} \\
-\infty \;\; &\text{永不需要}(j > i \text{ 被 mask}) \\
\star \;\; &\text{本步新算,仅此一行}
\end{aligned}
$$

三件事一目了然:

1. 左上 $3\times3$ 一个元素都没变——旧元素是 $q_i \cdot k_j\;(i, j \le 3)$,不含 $q_4, k_4, v_4$
2. 新增第 4 列的上面 3 个格子全被 mask($j = 4 > i$),永不计算
3. 本步只算第 4 行的 4 个点积 + 该行 softmax + $o_4$ 一个加权和

### 6.2 一般情形

**本步要算的:**

$$
x_{n+1} = \mathrm{wte}[t_{n+1}] + \mathrm{wpe}[n]
$$

每层先算新 token 自己的三个向量:

$$
q_{n+1} = \mathrm{LN}(x_{n+1})\,W_Q \;\;(\text{用完即弃}), \qquad
k_{n+1} = \mathrm{LN}(x_{n+1})\,W_K, \qquad
v_{n+1} = \mathrm{LN}(x_{n+1})\,W_V \;\;(\text{追加进缓存})
$$

新行的 $n+1$ 个得分(每个是一次 $d$ 维点积):

$$
s_{n+1,\,j} = \frac{q_{n+1} \cdot k_j}{\sqrt{d}}, \qquad j = 1, 2, \dots, n+1
\qquad \Longrightarrow \;\; \text{必须读 } k_1, k_2, \dots, k_n
$$

新行 softmax 与新 token 的输出:

$$
a_{n+1,\,j} = \frac{e^{s_{n+1,\,j}}}{\sum_{l=1}^{n+1} e^{s_{n+1,\,l}}}
$$

$$
\boxed{\;o_{n+1} = \sum_{j=1}^{n+1} a_{n+1,\,j}\, v_j = \underbrace{a_{n+1,1}\,v_1 + \cdots + a_{n+1,n}\,v_n}_{\text{必须读 } v_1,\dots,v_n} + \;a_{n+1,n+1}\,v_{n+1}\;}
$$

之后:12 个头的 $o_{n+1}$ 拼接 $\to$ 乘 $W_{\text{proj}}$ 加 bias $\to$ 残差相加 $\to$ MLP(逐 token,不读任何历史向量)$\to$ 残差相加。

**本步不用算的:**

| 不用算 | 原因(从下标读出) |
|---|---|
| 所有 $s_{ij},\, a_{ij},\, o_i \;(i \le n)$ | 只含 $q_i,\, k_1..k_i,\, v_1..v_i$,下标 $\le n < n+1$,与 $t_{n+1}$ 无关 |
| $s_{i,\,n+1} = q_i \cdot k_{n+1} \;(i \le n)$ | $j = n+1 > i$,被 mask,永不需要 |
| $q_1, \dots, q_n$ | 只出现在第 $1..n$ 行,而那些行不再算——**Q 用完即弃,从不缓存** |

**本步必须读的旧向量(每层每头):**

$$
k_1, k_2, \dots, k_n \quad\text{(算新行得分)}, \qquad v_1, v_2, \dots, v_n \quad\text{(算加权和)}
$$

全模型合计 $2 L n d$ 个数;fp16 下每步约 $2 \times 12 \times n \times 768 \times 2$ 字节($n = 1024$ 时 $\approx 37.7$ MB)。再加全部模型权重($\approx 124\text{M}$ 参数,fp16 $\approx 249$ MB/步)——这才是 decode 带宽受限的大头(见 Day 06)。

## 7. 为什么旧的 $k$、$v$ 可以原样复用(下标归纳)

对任意 $i \le n$:$o_i = \sum_{j \le i} a_{ij} v_j$ 中 $j \le i \le n$,新 token 的 $k_{n+1}, v_{n+1}$ 不出现。沿层归纳:

$$
x_i^{(0)} = \mathrm{wte}[t_i] + \mathrm{wpe}[i-1] \;\;\Rightarrow\;\; \text{只依赖 } t_i
$$

$$
x_i^{(\ell-1)} \text{ 只依赖 } t_1..t_i
\;\Rightarrow\;
\underbrace{\sum_{j \le i} a_{ij} v_j}_{\text{下标 } j \le i} + \text{MLP(逐 token)}
\;\Rightarrow\;
x_i^{(\ell)} \text{ 只依赖 } t_1..t_i
$$

$$
\therefore\;\; k_i = \mathrm{LN}(x_i)\,W_K,\;\; v_i = \mathrm{LN}(x_i)\,W_V \;\;\text{只依赖 } t_1..t_i \text{,且权重固定}
$$

第 $n+1$ 步读到的 $k_i, v_i$ 与第 $i$ 步算出的逐位相同——**复用无损** $\blacksquare$

KV cache 的定义由此而来:每层每头保存 $k_1, \dots, k_n$ 和 $v_1, \dots, v_n$,每步各追加 $k_{n+1}, v_{n+1}$。

## 8. 计算量对比(每层每头,$m = n+1$)

无缓存(每步重算整个矩阵):

$$
\underbrace{2m^2 d}_{\text{点积 } s_{ij}} + \underbrace{2m^2 d}_{\text{加权和 } o_i} + \underbrace{6md^2}_{\text{QKV 投影}} \;=\; 4m^2 d + 6md^2
\qquad (\text{Day 06 的 } 4N^2d + 3N^2 \text{ 即此处注意力两项})
$$

有缓存(只算新行):

$$
\underbrace{2(n{+}1)d}_{\text{点积}} + \underbrace{2(n{+}1)d}_{\text{加权和}} + \underbrace{6d^2}_{\text{QKV 投影,仅 1 个 token}} \;=\; 4(n{+}1)d + 6d^2
$$

注意力部分之比 $\approx n$:序列越长省得越多($n = 1024$ 时理论约 $1000\times$;Day 05 实测仅 $\sim 2.6\times$,是小矩阵下框架开销占主导)。

## 9. 结论

| 工程结论 | 逐元素出处 |
|---|---|
| 新 token 只算一行 | 本步新增元素只有 $s_{n+1,j}\;(j \le n{+}1)$ 和 $o_{n+1}$ |
| 必须读全部历史 $k, v$ | $s_{n+1,j} = q_{n+1} \cdot k_j$ 与 $o_{n+1} = \sum_j a_{n+1,j} v_j$ 的下标 $j$ 跑遍 $1..n$ |
| 旧结果永不变 | $o_i = \sum_{j \le i} a_{ij} v_j$ 中 $j \le i \le n$,新 token 的 $k, v$ 不出现 |
| 新列永不计算 | $s_{i,n+1}$ 的 $j = n{+}1 > i$,被 mask |
| Q 不缓存 | 新行只用 $q_{n+1}$;$q_1..q_n$ 只在不再计算的旧行里出现 |
