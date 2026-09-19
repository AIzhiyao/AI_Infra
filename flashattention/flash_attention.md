# FlashAttention 原理与前后向过程

## 1. 标准 Attention 为什么需要将中间结果写入 HBM？

标准缩放点积 Attention 为：

$$
S = \tau QK^\top, \qquad
P = \operatorname{softmax}(S), \qquad
O = PV,
$$

其中 $\tau = 1 / \sqrt{d}$。在朴素实现中，$S$ 和 $P$ 通常需要写入高带宽显存（High Bandwidth Memory，HBM），主要有以下两个原因。

### 1.1 中间结果需要跨 Kernel 传递

$QK^\top$、Softmax 和 $PV$ 往往由多个独立的 GPU Kernel 依次完成。寄存器和共享内存中的数据只在当前 Kernel 的生命周期内有效，因此前一个 Kernel 必须先将输出写回 HBM，后一个 Kernel 才能读取并继续计算。

### 1.2 寄存器无法跨 Thread Block 生命周期保存数据

一个 Grid 中的 Thread Block 数量通常远多于 GPU 同一时刻能够驻留的数量。Block 执行结束后，其寄存器会立即被回收并重新分配；若结果尚未写回 HBM，数据就会丢失。由于 Block 的调度时间及其所在的流式多处理器（Streaming Multiprocessor，SM）由硬件动态决定，Kernel 不能依赖寄存器跨 Block 或跨 Kernel 保存状态。

### 1.3 由此产生的访存开销

完整的 $S$ 和 $P$ 都是 $N \times N$ 矩阵。随着序列长度 $N$ 增长，读写这两个矩阵会产生 $O(N^2)$ 的额外 HBM 访问量。FlashAttention 的核心目标正是避免在 HBM 中完整物化它们。

---

## 2. 寄存器溢出

当线程所需的寄存器数量超过可用容量时，就会发生寄存器溢出（Register Spill）。编译器会将溢出的数据转存到线程私有的 Local Memory。Local Memory 在物理上仍位于全局内存中，虽然访问可能经过 L1/L2 Cache，但延迟仍远高于寄存器。

常见触发原因包括：

- Tile 过大，使单个线程需要持有过多累加器；
- Kernel 中同时存活的中间变量过多；
- 为提高占用率（Occupancy），限制了每个线程可用的寄存器数量。

寄存器溢出会增加全局内存流量，可能抵消分块和算子融合带来的性能收益。

---

## 3. FlashAttention 的核心思想

FlashAttention 通过分块（Tiling）和在线 Softmax（Online Softmax）在片上逐块完成 Attention 计算。每个得分块 $S_{ij}$ 及其指数结果用完即可覆盖，无须写入 HBM。

### 3.1 符号约定

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| $N$ | 序列长度 | 标量 |
| $d$ | 注意力头维度（Head Dimension） | 标量 |
| $Q, K, V$ | Query、Key、Value | $N \times d$ |
| $B_r$ | Query Tile 的行数 | 标量 |
| $B_c$ | Key/Value Tile 的行数 | 标量 |
| $Q_i$ | 第 $i$ 个 Query Tile | $B_r \times d$ |
| $K_j, V_j$ | 第 $j$ 个 Key/Value Tile | $B_c \times d$ |
| $S_{ij}$ | 第 $(i,j)$ 个得分 Tile | $B_r \times B_c$ |
| $m, \ell$ | 每个 Query 行的在线最大值与指数和 | $B_r$ |
| $O$ | 当前 Query Tile 的未归一化输出累加器 | $B_r \times d$ |

本文约定：向量与矩阵相乘时，向量按行广播；$\odot$ 表示逐元素乘法。

### 3.2 分块大小

设每个 SM 可用于当前计算的片上 SRAM 容量为 $M$，并以可容纳的标量元素个数计量。前向传播和反向传播采用相同的分块方式：

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil,
\qquad
B_r = \min\left(
    \left\lceil \frac{M}{4d} \right\rceil,
    d
\right).
$$

对应的 Tile 数量为：

$$
T_r = \left\lceil \frac{N}{B_r} \right\rceil,
\qquad
T_c = \left\lceil \frac{N}{B_c} \right\rceil.
$$

实际实现会根据数据类型、寄存器数量、共享内存容量、Tensor Core 指令形状和 Occupancy 调整 $B_r$ 与 $B_c$，不一定直接采用上述理论值。

---

## 4. 前向传播过程

### 4.1 分块与并行映射

前向传播沿用第 3.2 节定义的 $B_r$、$B_c$、$T_r$ 和 $T_c$。其中，$Q$ 按行划分为 $T_r$ 个块，$K$ 和 $V$ 按行划分为 $T_c$ 个块。

- 每个 Thread Block 负责一个 Query 块 $Q_i$；
- 各个 Thread Block 并行执行；
- 每个 Block 独立处理对应的 Query 行，彼此之间不存在依赖。

### 4.2 Thread Block 初始化

#### 4.2.1 加载 Query 块

将当前 Block 负责的 $Q_i \in \mathbb{R}^{B_r \times d}$ 从 HBM 加载到 SRAM，并使其在整个内层循环期间驻留在片上。

#### 4.2.2 初始化片上状态

在寄存器或部分 SRAM 中，为当前 Query 块的每一行初始化以下状态：

$$
m = (-\infty)_{B_r},
\qquad
\ell = (0)_{B_r},
\qquad
O = (0)_{B_r \times d}.
$$

其中，$m$ 是行最大值，$\ell$ 是行和，$O$ 是尚未除以 $\ell$ 的**未归一化**输出累加器。

### 4.3 内层循环：遍历所有 Key/Value 块

对于 $j = 1, 2, \ldots, T_c$，依次执行以下操作。

#### 4.3.1 加载 Key/Value 块

将当前的 $K_j, V_j \in \mathbb{R}^{B_c \times d}$ 从 HBM 加载到 SRAM，并覆盖上一轮的 $K_{j-1}$ 和 $V_{j-1}$。

#### 4.3.2 计算缩放点积得分

在 SRAM 中使用 Tensor Core 计算缩放点积：

$$
S_{ij} = \tau \cdot \left(Q_i K_j^T\right),
$$

其中，$\tau = 1/\sqrt{d}$，结果 $S_{ij} \in \mathbb{R}^{B_r \times B_c}$ 保存在 SRAM 中。

#### 4.3.3 应用掩码（可选）

如果存在 Causal Mask 或 Padding Mask，则将 $S_{ij}$ 中对应的位置设为 $-\infty$，或设为一个足够大的负数。整个掩码操作在 SRAM 中完成。

#### 4.3.4 计算局部行统计量

首先并行计算 $S_{ij}$ 每一行的局部最大值：

$$
\widetilde{m}_{ij}
= \operatorname{rowmax}(S_{ij}),
$$

其中，$\widetilde{m}_{ij}$ 的长度为 $B_r$，可通过 Warp 内线程协作完成归约。

然后计算减去局部最大值后的指数项：

$$
\widetilde{P}_{ij}
= \exp\!\left(S_{ij} - \widetilde{m}_{ij}\right),
$$

结果 $\widetilde{P}_{ij} \in \mathbb{R}^{B_r \times B_c}$ 保存在 SRAM 中。接着计算局部行和：

$$
\widetilde{\ell}_{ij}
= \operatorname{rowsum}\!\left(\widetilde{P}_{ij}\right),
$$

其中，$\widetilde{\ell}_{ij}$ 的长度为 $B_r$。

#### 4.3.5 在线更新全局统计量

首先逐元素计算新的全局最大值：

$$
m_{\mathrm{new}}
= \max\!\left(m, \widetilde{m}_{ij}\right).
$$

再逐元素计算两个缩放因子：

$$
\alpha = \exp\!\left(m - m_{\mathrm{new}}\right),
\qquad
\beta = \exp\!\left(\widetilde{m}_{ij} - m_{\mathrm{new}}\right).
$$

最后逐元素更新全局行和：

$$
\ell
\leftarrow
\alpha \odot \ell
+ \beta \odot \widetilde{\ell}_{ij}.
$$

上述向量操作均在寄存器中完成，无须写回 HBM。

#### 4.3.6 更新未归一化输出累加器

将历史累加结果的缩放与当前 Key/Value 块的贡献合并为一次更新：

$$
O
\leftarrow
\alpha \odot O
+ \beta \odot \left(\widetilde{P}_{ij}V_j\right).
$$

其中，第一项重新缩放此前累积的输出，第二项表示当前 Key/Value 块的贡献。更新后的 $O$ 仍是未归一化结果。

#### 4.3.7 更新局部状态

更新全局最大值：

$$
m \leftarrow m_{\mathrm{new}}.
$$

$\ell$ 已在前一步完成更新；$S_{ij}$ 和 $\widetilde{P}_{ij}$ 可在下一轮循环中直接覆盖。

### 4.4 最终归一化与写回

处理完全部 $T_c$ 个 Key/Value 块后，在寄存器中对每一行执行一次除法：

$$
O_{\mathrm{final}} = O / \ell.
$$

由此得到归一化后的 Attention 输出。

- 训练时如有需要，将最终的 $\ell$ 以及可能需要的 $m$ 写回 HBM，供反向传播使用；
- 将 $O_{\mathrm{final}} \in \mathbb{R}^{B_r \times d}$ 从 SRAM 或寄存器写回 HBM 中的对应输出行。

### 4.5 多级并行与硬件协作

- **Grid 级**：多个 Thread Block 并行执行，每个 Block 处理不同的 Query 块；
- **Block 级**：单个 Thread Block 顺序遍历所有 Key/Value 块，每个块的计算由整个 Thread Block 协作完成；
- **Warp 级**：不同 Warp 负责输出 Tile 的不同部分，并使用 Tensor Core 的 MMA 指令完成矩阵乘法；
- **寄存器**：保存各线程负责的 $O$ 累加器片段，以及 $m$、$\ell$ 等标量或向量片段；
- **SRAM**：保存 $Q_i$、当前的 $K_j$ 和 $V_j$，以及中间矩阵 $S_{ij}$、$\widetilde{P}_{ij}$；
- **HBM**：负责初始加载 $Q$、$K$、$V$ 和最终写回 $O$；训练时还会保存 $\ell$、$m$，但不会保存中间矩阵 $S$ 或 $P$。

### 4.6 关键优化点

- **中间矩阵不落地**：$S$ 和 $P$ 只在 SRAM 中生成和使用，不写回 HBM；
- **未归一化累加器**：内层循环不执行除法，仅在循环结束后进行一次归一化；
- **Tensor Core 高效计算**：两次矩阵乘法均使用 Tensor Core，以发挥混合精度计算的吞吐优势；
- **数据复用**：$Q_i$ 常驻 SRAM，供所有 Key/Value 块复用；$K$、$V$ 块还可能通过 L2 Cache 被不同 SM 共享。

---

## 5. Online Softmax 的正确性

假设已经处理了部分 Key，当前状态为 $(m, \ell, O)$，并满足：

$$
\ell = \sum_{t \in \mathcal{A}} \exp(s_t - m),
$$

$$
O = \sum_{t \in \mathcal{A}} \exp(s_t - m)v_t,
$$

其中，$\mathcal{A}$ 表示已经处理的 Key 集合。

对于新读入的 Tile，其局部状态满足：

$$
\widetilde{\ell}
= \sum_{t \in \mathcal{B}}
\exp(s_t - \widetilde{m}),
$$

$$
\widetilde{O}
= \sum_{t \in \mathcal{B}}
\exp(s_t - \widetilde{m})v_t.
$$

令：

$$
m_{\mathrm{new}} = \max(m, \widetilde{m}),
$$

则旧状态和新 Tile 都可以转换到以 $m_{\mathrm{new}}$ 为基准的同一指数尺度：

$$
\ell_{\mathrm{new}}
= \exp(m - m_{\mathrm{new}})\ell
+ \exp(\widetilde{m} - m_{\mathrm{new}})\widetilde{\ell},
$$

$$
O_{\mathrm{new}}
= \exp(m - m_{\mathrm{new}})O
+ \exp(\widetilde{m} - m_{\mathrm{new}})\widetilde{O}.
$$

处理完所有 Tile 后，有：

$$
\frac{O}{\ell}
= \sum_t
\frac{\exp(s_t)}{\sum_u \exp(s_u)}v_t,
$$

该结果与对完整得分矩阵执行一次标准 Softmax 完全等价。

因此，$S_{ij}$ 和 $\widetilde{P}_{ij}$ 只是可以随时覆盖的临时数据，而 $m$、$\ell$ 和 $O$ 则以压缩形式保留了此前所有 Tile 的全局状态。

---

## 6. 反向传播

> **算法 4：FlashAttention 反向传播**

### 6.1 输入条件

以下矩阵均存储在 HBM 中：

$$
\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO}
\in \mathbb{R}^{N \times d}.
$$

前向传播保存的行级统计量也存储在 HBM 中：

$$
\ell, m \in \mathbb{R}^N.
$$

算法还需要以下输入：

- 片上 SRAM 大小 $M$；
- Softmax 缩放常数 $\tau \in \mathbb{R}$；
- 掩码函数 `MASK`；
- Dropout 概率 $p_{\text{drop}}$；
- 前向传播保存的伪随机数生成器状态 $\mathcal{R}$。

### 6.2 初始化与分块

#### 6.2.1 恢复随机状态

首先，将伪随机数生成器恢复到前向传播保存的状态：

$$
\operatorname{RNGState} \leftarrow \mathcal{R}.
$$

由此可在反向传播中重新生成与前向传播一致的 Dropout 掩码。

#### 6.2.2 划分输入、输出与行级统计量

反向传播沿用第 3.2 节定义的分块方式：将 $\mathbf{Q}$、$\mathbf{O}$ 和 $\mathbf{dO}$ 分别划分为 $T_r$ 个块，将 $\mathbf{K}$ 和 $\mathbf{V}$ 分别划分为 $T_c$ 个块。其中：

$$
\mathbf{O}_i, \mathbf{dO}_i
\in \mathbb{R}^{B_r \times d},
\qquad
1 \leq i \leq T_r.
$$

将 $\ell$ 和 $m$ 各自划分为 $T_r$ 个长度为 $B_r$ 的块：

$$
\ell_1, \ldots, \ell_{T_r},
\qquad
m_1, \ldots, m_{T_r}.
$$

#### 6.2.3 初始化并划分输入梯度

在 HBM 中初始化：

$$
\mathbf{dQ} = (0)_{N \times d},
\qquad
\mathbf{dK} = (0)_{N \times d},
\qquad
\mathbf{dV} = (0)_{N \times d}.
$$

将 $\mathbf{dQ}$ 划分为 $T_r$ 个 $B_r \times d$ 的块：

$$
\mathbf{dQ}_1, \ldots, \mathbf{dQ}_{T_r}.
$$

$\mathbf{dK}$ 和 $\mathbf{dV}$ 分别划分为 $T_c$ 个 $B_c \times d$ 的块：

$$
\mathbf{dK}_1, \ldots, \mathbf{dK}_{T_c},
\qquad
\mathbf{dV}_1, \ldots, \mathbf{dV}_{T_c}.
$$

### 6.3 外层循环：遍历 Key/Value Tile

对于 $1 \leq j \leq T_c$，依次处理每个 Key/Value Tile。

#### 6.3.1 加载 $K_j$ 和 $V_j$

将当前的 $\mathbf{K}_j, \mathbf{V}_j \in \mathbb{R}^{B_c \times d}$ 从 HBM 加载到片上 SRAM。

#### 6.3.2 初始化当前 Tile 的梯度累加器

在 SRAM 中初始化对应的梯度累加器：

$$
\widetilde{\mathbf{dK}}_j = (0)_{B_c \times d},
\qquad
\widetilde{\mathbf{dV}}_j = (0)_{B_c \times d}.
$$

### 6.4 内层循环：遍历 Query Tile

对于 $1 \leq i \leq T_r$，依次处理每个 Query Tile。

#### 6.4.1 加载当前 Query Tile 的状态

将以下数据从 HBM 加载到片上 SRAM：

$$
\mathbf{Q}_i,
\mathbf{O}_i,
\mathbf{dO}_i,
\mathbf{dQ}_i,
\ell_i,
m_i.
$$

#### 6.4.2 重算得分与 Softmax 概率

首先，在片上重新计算缩放点积得分：

$$
\mathbf{S}_{ij}
= \tau \mathbf{Q}_i \mathbf{K}_j^T
\in \mathbb{R}^{B_r \times B_c}.
$$

随后应用掩码函数：

$$
\mathbf{S}^{\text{masked}}_{ij}
= \text{MASK}(\mathbf{S}_{ij}).
$$

最后，使用前向传播保存的 $\ell_i$ 和 $m_i$ 重建当前概率块：

$$
\mathbf{P}_{ij}
= \text{diag}(\ell_i)^{-1}
  \exp\!\left(
      \mathbf{S}^{\text{masked}}_{ij} - m_i
  \right)
\in \mathbb{R}^{B_r \times B_c}.
$$

#### 6.4.3 重建并应用 Dropout 掩码

利用保存的伪随机数生成器状态，重建与前向传播一致的 Dropout 掩码：

$$
\mathbf{Z}_{ij} \in \mathbb{R}^{B_r \times B_c}.
$$

其中，每个元素的取值为：

$$
(\mathbf{Z}_{ij})_{ab}
=
\begin{cases}
\dfrac{1}{1-p_{\text{drop}}},
& \text{以概率 } 1-p_{\text{drop}}, \\
0,
& \text{以概率 } p_{\text{drop}}.
\end{cases}
$$

然后将 Dropout 掩码应用于概率块：

$$
\mathbf{P}^{\text{dropped}}_{ij}
= \mathbf{P}_{ij} \circ \mathbf{Z}_{ij}.
$$

#### 6.4.4 累加 $\mathbf{dV}_j$

$$
\widetilde{\mathbf{dV}}_j
\leftarrow
\widetilde{\mathbf{dV}}_j
+ \left(\mathbf{P}^{\text{dropped}}_{ij}\right)^\top
  \mathbf{dO}_i,
\qquad
\widetilde{\mathbf{dV}}_j
\in \mathbb{R}^{B_c \times d}.
$$

#### 6.4.5 计算概率梯度

先计算 Dropout 后概率对应的梯度：

$$
\widetilde{\mathbf{dP}}^{\text{dropped}}_{ij}
= \mathbf{dO}_i \mathbf{V}_j^\top
\in \mathbb{R}^{B_r \times B_c}.
$$

再反向通过 Dropout：

$$
\widetilde{\mathbf{dP}}_{ij}
= \widetilde{\mathbf{dP}}^{\text{dropped}}_{ij}
  \circ \mathbf{Z}_{ij}.
$$

#### 6.4.6 计算 Softmax 得分梯度

首先，计算每个 Query 行对应的辅助量：

$$
D_i
= \text{rowsum}\!\left(
    \mathbf{dO}_i \circ \mathbf{O}_i
  \right)
\in \mathbb{R}^{B_r}.
$$

然后计算 Softmax 得分梯度：

$$
\widetilde{\mathbf{dS}}_{ij}
= \mathbf{P}_{ij}
  \circ \left(
      \widetilde{\mathbf{dP}}_{ij} - D_i
  \right)
\in \mathbb{R}^{B_r \times B_c}.
$$

其中，$D_i$ 沿列方向广播。

#### 6.4.7 累加输入梯度

先累加当前 Query Tile 的梯度：

$$
\mathbf{dQ}_i
\leftarrow
\mathbf{dQ}_i
+ \tau \widetilde{\mathbf{dS}}_{ij}\mathbf{K}_j,
\qquad
\mathbf{dQ}_i \in \mathbb{R}^{B_r \times d}.
$$

计算完成后，将更新后的 $\mathbf{dQ}_i$ 写回 HBM。

再累加当前 Key Tile 的梯度：

$$
\widetilde{\mathbf{dK}}_j
\leftarrow
\widetilde{\mathbf{dK}}_j
+ \tau \widetilde{\mathbf{dS}}_{ij}^\top \mathbf{Q}_i,
\qquad
\widetilde{\mathbf{dK}}_j
\in \mathbb{R}^{B_c \times d}.
$$

### 6.5 写回结果

处理完所有 Query Tile 后，结束当前 $j$ 对应的内层循环。

#### 6.5.1 写回当前 Key/Value Tile 的梯度

将当前 Key/Value Tile 对应的片上累加结果写回 HBM：

$$
\mathbf{dK}_j \leftarrow \widetilde{\mathbf{dK}}_j,
\qquad
\mathbf{dV}_j \leftarrow \widetilde{\mathbf{dV}}_j.
$$

处理完所有 Key/Value Tile 后，结束外层循环。

#### 6.5.2 返回梯度

最终返回：

$$
\mathbf{dQ},
\qquad
\mathbf{dK},
\qquad
\mathbf{dV}.
$$
