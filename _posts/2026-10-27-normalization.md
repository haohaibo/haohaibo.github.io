---
layout: post
title: Normalization
date: 2026-10-27 09:00:00
description: Notes and research on normalization - BatchNorm, LayerNorm and RMSNorm.
tags: [notes and research, normalization, LLM]
toc:
  sidebar: left
---

本文主要讲解 BatchNorm、LayerNorm 和 RMSNorm 的区别与联系。

三者的区别主要体现在统计量的计算维度和归一化方式上：

1. **和哪些数一起计算统计量？**——决定了 BatchNorm 和 LayerNorm 的区别。
2. **如何调整这些数？**——决定了 LayerNorm 和 RMSNorm 的区别。

> **BatchNorm：同一个特征，跨样本做标准化。**
>
> **LayerNorm：同一个样本，跨特征做标准化。**
>
> **RMSNorm：通常沿 LayerNorm 相同的维度，但不减均值，只控制整体尺度。**

## 一、为什么需要 Norm？

### 1. 深层网络中的数值尺度容易发生变化

神经网络会反复执行这样的计算：

```text
输入 → 线性变换 → 激活函数 → 线性变换 → …… → 输出
```

每次线性变换、非线性操作和残差相加，都可能改变隐藏表示的数值尺度。

例如，两个隐藏向量：

$$
\begin{aligned}
\mathbf{x}^{(1)} &= [0.1,\ 0.2,\ 0.3,\ 0.4], \\
\mathbf{x}^{(2)} &= [10,\ 20,\ 30,\ 40].
\end{aligned}
$$

它们的方向相同，但整体尺度相差 100 倍。把它们送入后续网络，可能产生很不一样的激活、注意力分数和梯度。

**Norm 的基本作用，是先把隐藏表示调整到一个较可控的尺度，再交给后续计算。**

注意：这里主要是在调整**激活值，也就是中间表示**，不是直接归一化权重。

### 2. 它能帮助优化更稳定

通常包括：

- 减少隐藏表示尺度波动对后续计算的影响。
- 改善训练过程中的数值和优化稳定性。
- 在一些架构中允许使用更大的学习率。
- 降低模型对部分参数尺度的敏感性。

但不能理解成：

> “加上 Norm，就不会梯度爆炸或消失。”

Norm 只是稳定训练的一部分，初始化、残差连接、学习率等仍然重要。

### 3. 不只是“让数据服从正态分布”

这是一个常见误解：**归一化不等于高斯化。**

减均值、除标准差，只是在调整均值和方差，并不会把任意分布变成正态分布，也不意味着输出一定在 $$[-1,1]$$ 内。

另外，BatchNorm 最初常用“减少内部协变量偏移”解释其作用；但后续研究表明，不能把它的有效性简单归结为这一点，优化性质的改善同样重要。

## 二、先看图：它们沿哪个维度计算？

假设输入是一张二维矩阵：

$$
X \in \mathbb{R}^{N\times D}.
$$

- $$N$$：样本数，也就是 batch size。
- $$D$$：每个样本的特征数。

那么：

- **BatchNorm：每列独立，跨行统计。**
- **LayerNorm：每行独立，跨列统计。**
- **RMSNorm：通常也每行独立，跨列统计。**

图中同色格子一起计算统计量：

{% include figure.liquid path="assets/img/norm-dimensions.svg" class="img-fluid" alt="BatchNorm、LayerNorm、RMSNorm 的统计维度" caption="BatchNorm、LayerNorm、RMSNorm 的统计维度" %}

这里必须区分两件事：

> **统计量的计算维度**，和**可学习参数的维度**，不是一回事。

例如，二维输入中的 LN：

- 每个样本有自己的均值、方差。
- 但可学习的 $$\gamma$$、$$\beta$$ 通常按特征设置，在样本之间共享。

## 三、BatchNorm：跨样本，标准化同一个特征

### 1. 计算公式

输入矩阵 $$X\in\mathbb{R}^{N\times D}$$，$$x_{n,d}$$ 表示第 $$n$$ 个样本的第 $$d$$ 个特征。

对每个特征 $$d$$，计算：

$$
\begin{aligned}
\mu_d &= \frac{1}{N}\sum_{n=1}^{N}x_{n,d}, \\
\sigma_d^2 &= \frac{1}{N}\sum_{n=1}^{N}(x_{n,d}-\mu_d)^2, \\
z_{n,d} &= \frac{x_{n,d}-\mu_d}{\sqrt{\sigma_d^2+\varepsilon}}, \\
y_{n,d} &= \gamma_d z_{n,d}+\beta_d.
\end{aligned}
$$

含义：

- $$\mu_d$$：这个特征在当前 batch 中的均值。
- $$\sigma_d^2$$：这个特征在当前 batch 中的方差。
- $$\varepsilon$$：防止分母为零、改善数值稳定性的小常数。
- $$\gamma_d$$、$$\beta_d$$：可学习的缩放和平移参数。

**注意，$$\varepsilon$$ 加在平方根里面。**

这里归一化公式使用的是除以 $$N$$ 的方差，而不是统计学中用于无偏估计的 $$N-1$$。

### 2. CNN 中的 BatchNorm

卷积输入通常是：

$$
X\in\mathbb{R}^{N\times C\times H\times W}.
$$

常规 BatchNorm 会：

> **固定通道 $$c$$，沿 $$N$$、$$H$$、$$W$$ 一起计算均值和方差。**

也就是，每个通道有一组统计量和一组 $$\gamma$$、$$\beta$$。

所以，“BatchNorm 只沿 batch 维计算”只是二维输入下的简化说法；在 CNN 中，它还会合并空间位置。

### 3. 训练与推理不同

这是 BatchNorm 最重要的特点之一。

#### 训练时

使用**当前 batch 的均值和方差**进行归一化，同时更新运行统计量：

$$
\begin{aligned}
\mu_{\mathrm{running}}
&\leftarrow (1-\alpha)\mu_{\mathrm{running}} \\
&\quad +\alpha\mu_{\mathrm{batch}}.
\end{aligned}
$$

运行方差也会更新，具体更新约定可能因框架而不同。

#### 推理时

通常使用训练过程中积累的：

```text
running_mean
running_var
```

而不是临时用当前推理 batch 计算统计量。

否则，同一个样本的输出可能随“和哪些样本一起推理”而变化。

### 4. 优点与局限

**优点：**

- 在许多 CNN 中效果很好。
- batch 统计量的随机波动可能带来一定正则化作用。

**局限：**

- 训练时，一个样本的输出依赖同一统计集合中的其他样本。
- 统计集合较小时，估计容易不稳定。
- 训练和推理的统计行为不一致。
- 分布式训练中，是否跨设备同步统计量会影响行为。

两个容易忽视的细节：

- **batch size 为 1，不一定意味着卷积 BN 无法使用**，因为还可能有 $$H\times W$$ 个空间位置参与统计。
- **梯度累积不自动等价于大 batch 的 BN**，因为 BN 通常在每次前向计算时分别统计。

## 四、LayerNorm：每个样本内部做标准化

### 1. 计算公式

对单个样本的特征向量：

$$
\mathbf{x}=[x_1,x_2,\ldots,x_D].
$$

计算：

$$
\begin{aligned}
\mu &= \frac{1}{D}\sum_{i=1}^{D}x_i, \\
\sigma^2 &= \frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2, \\
z_i &= \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}, \\
y_i &= \gamma_i z_i+\beta_i.
\end{aligned}
$$

与 BatchNorm 相比，操作形式非常接近，主要变化是：

> **均值和方差不再跨样本计算，而是在当前样本的特征内部计算。**

忽略 $$\varepsilon$$，且方差非零时，归一化核心 $$\mathbf{z}$$ 满足：

$$
\begin{aligned}
\operatorname{mean}(\mathbf{z}) &= 0, \\
\operatorname{Var}(\mathbf{z}) &= 1.
\end{aligned}
$$

加入 $$\varepsilon$$ 后，方差只是接近 1。

### 2. Transformer 中的 LayerNorm

Transformer 的隐藏表示通常是：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

- $$B$$：batch size。
- $$T$$：序列长度。
- $$D$$：隐藏维度。

常规 LayerNorm 固定 $$(b,t)$$，只沿 $$D$$ 计算：

```text
第 b 个样本：

token 1 → [D 个特征] → 单独计算 μ、σ²
token 2 → [D 个特征] → 单独计算 μ、σ²
token 3 → [D 个特征] → 单独计算 μ、σ²
```

因此：

- 不把不同样本混在一起。
- 不把不同 token 混在一起。
- 每个 token 都有自己的均值和方差。
- $$\gamma$$、$$\beta$$ 通常在所有样本、所有 token 之间共享。

**LayerNorm 这个名字，不代表它必然把“整层所有元素”一起归一化。** 实际归一化哪些维度，要看配置；Transformer 中通常是最后的隐藏维度。

### 3. 训练与推理

LN 在训练和推理时，通常都使用**当前输入自身的统计量**。

没有 BatchNorm 那样的运行均值、运行方差。

因此，它很适合 batch 大小变化、变长序列等场景。

## 五、RMSNorm：不减均值，只调整均方根

RMS 是 **Root Mean Square，均方根**。

### 1. 计算公式

对一个 $$D$$ 维向量：

$$
\begin{aligned}
\operatorname{RMS}(\mathbf{x})
&=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2}, \\
z_i&=\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\varepsilon}}, \\
y_i&=\gamma_i z_i.
\end{aligned}
$$

标准 RMSNorm 通常只有可学习缩放 $$\gamma$$，没有 $$\beta$$；某些实现可能加入额外偏置。

和 LN 相比：

- 不计算用于中心化的均值。
- 不执行 $$x_i-\mu$$。
- 分母使用**平方的平均值**，而不是**减均值后的平方偏差平均值**。

所以它不是简单地：

> “把 LN 分子中的减均值删掉，其他保持不变。”

**分母也变了。**

### 2. 它保证什么？

忽略 $$\varepsilon$$，对于非零向量：

$$
\frac{1}{D}\sum_{i=1}^{D}z_i^2=1
\quad\Longleftrightarrow\quad
\operatorname{RMS}(\mathbf{z})=1.
$$

但它**不保证**：

$$
\begin{aligned}
\operatorname{mean}(\mathbf{z}) &= 0, \\
\operatorname{Var}(\mathbf{z}) &= 1.
\end{aligned}
$$

这就是 RMSNorm 与 LayerNorm 最核心的区别。

### 3. 与 L2 归一化的关系

$$
\begin{aligned}
\lVert\mathbf{x}\rVert_2&=\sqrt{\sum_{i=1}^{D}x_i^2}, \\
\operatorname{RMS}(\mathbf{x})&=\frac{\lVert\mathbf{x}\rVert_2}{\sqrt{D}}.
\end{aligned}
$$

因此，忽略 $$\varepsilon$$ 和可学习参数：

$$
\mathbf{z}=\sqrt{D}\,\frac{\mathbf{x}}{\lVert\mathbf{x}\rVert_2}.
$$

普通 L2 归一化把向量长度调整到 1；RMSNorm 把向量长度调整到 $$\sqrt{D}$$，使每个坐标的均方尺度为 1。

## 六、同一组数字，具体算一次

取：

$$
\mathbf{x}=[1,2,3,4].
$$

暂时忽略 $$\varepsilon$$，并设 $$\gamma=1$$、$$\beta=0$$。

### 1. LayerNorm

$$
\begin{aligned}
\mu&=2.5, \\
\sigma^2&=\frac{(-1.5)^2+(-0.5)^2+0.5^2+1.5^2}{4} \\
&=1.25, \\
\sigma&\approx1.118.
\end{aligned}
$$

所以：

$$
\operatorname{LN}(\mathbf{x})
\approx[-1.342,-0.447,0.447,1.342].
$$

结果围绕 0 分布。

### 2. RMSNorm

$$
\begin{aligned}
\operatorname{RMS}(\mathbf{x})^2
&=\frac{1^2+2^2+3^2+4^2}{4} \\
&=7.5, \\
\operatorname{RMS}(\mathbf{x})&\approx2.739.
\end{aligned}
$$

所以：

$$
\operatorname{RMSNorm}(\mathbf{x})
\approx[0.365,0.730,1.095,1.461].
$$

原来全部为正，归一化核心仍然全部为正：

{% include figure.liquid path="assets/img/layernorm-vs-rmsnorm.svg" class="img-fluid" alt="LayerNorm 与 RMSNorm 的数值对比" caption="LayerNorm 与 RMSNorm 的数值对比" %}

### 3. BatchNorm 要看一批样本

现在取两个样本：

$$
X=\begin{bmatrix}
1&2&3&4 \\
10&20&30&40
\end{bmatrix}.
$$

每列分别计算：

$$
\begin{aligned}
\boldsymbol{\mu}&=[5.5,11,16.5,22], \\
\boldsymbol{\sigma}&=[4.5,9,13.5,18].
\end{aligned}
$$

于是：

$$
\operatorname{BN}(X)=\begin{bmatrix}
-1&-1&-1&-1 \\
1&1&1&1
\end{bmatrix}.
$$

对比同一个矩阵：

| 方法    | 第一个样本的输出                | 第二个样本的输出 |
| ------- | ------------------------------- | ---------------- |
| BN      | $$[-1,-1,-1,-1]$$               | $$[1,1,1,1]$$    |
| LN      | $$[-1.342,-0.447,0.447,1.342]$$ | 与第一个样本相同 |
| RMSNorm | $$[0.365,0.730,1.095,1.461]$$   | 与第一个样本相同 |

为什么 LN、RMSNorm 两行相同？

因为第二个向量只是第一个向量乘以正数 10，而这两种归一化的核心都会消除这种整体缩放。

## 七、LayerNorm 与 RMSNorm 到底是什么关系？

### 1. 一个关键恒等式

先明确 $$\operatorname{RMS}(\mathbf{x})^2$$ 是什么。按 Root Mean Square 的计算顺序，先平方、再平均、最后开根号；平方之后根号被抵消，剩下的就是**平方的平均值**：

$$
\operatorname{RMS}(\mathbf{x})^2=\frac{1}{D}\sum_{i=1}^{D}x_i^2.
$$

它和方差的差别只在参照点：

| 量                                   | 计算方式             | 衡量什么                      |
| ------------------------------------ | -------------------- | ----------------------------- |
| $$\operatorname{RMS}(\mathbf{x})^2$$ | 直接平方，再平均     | 各坐标距离 $$0$$ 的平方平均   |
| $$\sigma^2$$                         | 减均值后平方，再平均 | 各坐标距离 $$\mu$$ 的平方平均 |

恒等式说的就是，这两个量只相差一个 $$\mu^2$$：

$$
\operatorname{RMS}(\mathbf{x})^2=\sigma^2+\mu^2,
$$

写开即

$$
\frac{1}{D}\sum_{i=1}^{D}x_i^2
=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2+\mu^2.
$$

#### 推导

使用除以 $$D$$ 的方差定义，从右边展开：

$$
\begin{aligned}
\sigma^2+\mu^2
&=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2+\mu^2 \\
&=\frac{1}{D}\sum_{i=1}^{D}\bigl(x_i^2-2\mu x_i+\mu^2\bigr)+\mu^2 \\
&=\frac{1}{D}\sum_{i=1}^{D}x_i^2
-\frac{2\mu}{D}\sum_{i=1}^{D}x_i
+\frac{1}{D}\sum_{i=1}^{D}\mu^2
+\mu^2 \\
&=\frac{1}{D}\sum_{i=1}^{D}x_i^2-2\mu^2+\mu^2+\mu^2 \\
&=\frac{1}{D}\sum_{i=1}^{D}x_i^2
=\operatorname{RMS}(\mathbf{x})^2.
\end{aligned}
$$

整个推导只有两处用到“$$\mu$$ 是均值”这一定义。

第一处，交叉项里的求和就是均值：

$$
\frac{1}{D}\sum_{i=1}^{D}x_i=\mu
\quad\Longrightarrow\quad
\frac{2\mu}{D}\sum_{i=1}^{D}x_i=2\mu\cdot\mu=2\mu^2.
$$

第二处，$$\mu^2$$ 与 $$i$$ 无关，求和后 $$D$$ 被约掉：

$$
\sum_{i=1}^{D}\mu^2=D\mu^2
\quad\Longrightarrow\quad
\frac{1}{D}\sum_{i=1}^{D}\mu^2=\mu^2.
$$

于是三个 $$\mu^2$$ 项恰好相消：

$$
-2\mu^2+\mu^2+\mu^2=0.
$$

**这个相消就是恒等式的全部内容。** 如果把 $$\mu$$ 换成任意常数 $$c$$，交叉项变成 $$2c\cdot\operatorname{mean}(\mathbf{x})$$，与其余项无法抵消——这也说明为什么参照点必须取均值。

#### 数值验证

对 $$\mathbf{x}=[1,2,3,4]$$：

$$
\begin{aligned}
\operatorname{RMS}(\mathbf{x})^2&=\frac{1+4+9+16}{4}=7.5, \\
\sigma^2&=1.25, \\
\mu^2&=2.5^2=6.25, \\
7.5&=1.25+6.25.
\end{aligned}
$$

这个分解可以这样读：

- $$\operatorname{RMS}(\mathbf{x})^2$$：相对 $$0$$ 的整体平方尺度。
- $$\sigma^2$$：围绕均值的波动。
- $$\mu^2$$：均值本身的偏移。

所以一个向量数值偏大，可能有两种原因：坐标之间差异大，即 $$\sigma^2$$ 大；或者坐标整体远离 $$0$$，即 $$\lvert\mu\rvert$$ 大。RMS 把两者一起算进去。

例如 $$\mathbf{x}=[5,5,5,5]$$ 完全没有波动，但并不接近 $$0$$：

$$
\sigma^2=0,\qquad
\operatorname{RMS}(\mathbf{x})^2=25=0+5^2.
$$

**所以“没有波动”不等于“整体尺度小”。**

#### 换成矩的语言

这条恒等式就是概率论里的经典结果：

$$
\operatorname{Var}(X)=\mathbb{E}[X^2]-\bigl(\mathbb{E}[X]\bigr)^2
\quad\Longleftrightarrow\quad
\mathbb{E}[X^2]=\operatorname{Var}(X)+\bigl(\mathbb{E}[X]\bigr)^2.
$$

- $$\operatorname{RMS}^2$$ 是**二阶原点矩**，以 $$0$$ 为参照。
- $$\sigma^2$$ 是**二阶中心矩**，以 $$\mu$$ 为参照。

两者相差的 $$\mu^2$$ 正是参照点平移的代价。LayerNorm 用中心矩，RMSNorm 用原点矩，区别恰好落在这一项上。

因此，它们的核心可以写成：

$$
\begin{aligned}
z_i^{\mathrm{LN}}&=\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}, \\
z_i^{\mathrm{RMS}}&=\frac{x_i}{\sqrt{\sigma^2+\mu^2+\varepsilon}}.
\end{aligned}
$$

区别清楚地落在两个地方：

|      | LayerNorm          | RMSNorm             |
| ---- | ------------------ | ------------------- |
| 分子 | 去掉共同均值       | 保留原坐标          |
| 分母 | 围绕均值的波动尺度 | 相对于 0 的整体尺度 |

**如果输入均值为 0，它们的归一化核心相同。** 完整模块还是否相同，则取决于 $$\gamma$$、$$\beta$$ 等参数设置。

### 2. LN 消除共同平移，RMSNorm 通常不消除

例如，$$\mathbf{1}$$ 表示全 1 向量：

$$
\begin{aligned}
\mathbf{x}&=[1,2,3,4], \\
\mathbf{x}+10\mathbf{1}&=[11,12,13,14].
\end{aligned}
$$

对 LN：

$$
\operatorname{LN}(\mathbf{x}+10\mathbf{1})
=\operatorname{LN}(\mathbf{x}).
$$

因为均值也增加了 10，减均值后完全抵消。

对 RMSNorm：

$$
\operatorname{RMSNorm}(\mathbf{x}+10\mathbf{1})
\ne\operatorname{RMSNorm}(\mathbf{x}).
$$

所以，忽略 $$\varepsilon$$ 时：

- **LN**：对正的整体缩放、共同平移都不敏感。
- **RMSNorm**：对正的整体缩放不敏感，但通常对共同平移敏感。

这里的“共同平移”指所有特征加上同一个数，不是每个特征随意加不同的数。

### 3. 一个极端例子

取：

$$
\mathbf{x}=[5,5,5,5].
$$

LN 因为减掉均值，归一化核心是：

$$
\mathbf{z}^{\mathrm{LN}}=[0,0,0,0].
$$

RMSNorm 忽略 $$\varepsilon$$ 后得到：

$$
\mathbf{z}^{\mathrm{RMS}}=[1,1,1,1].
$$

这说明：

> LN 会移除“所有特征共同抬高”的分量；RMSNorm 不会显式移除这个分量。

### 4. 几何上怎么理解？

暂不考虑 $$\varepsilon$$ 和可学习参数：

**LayerNorm：** 原向量 → 减均值，投影到“坐标之和为 0”的平面 → 把非零向量长度调整为 $$\sqrt{D}$$。

**RMSNorm：** 原向量 → 保持原方向 → 把长度调整为 $$\sqrt{D}$$。

因此，**RMSNorm 的核心保留原向量方向，LN 通常会改变原向量方向。**

加入逐特征 $$\gamma$$ 后，完整 RMSNorm 模块也可能改变方向。

## 八、为什么还需要可学习的 γ、β？

归一化核心是在设置一个统一的统计尺度，但网络可能需要不同特征具有不同的幅度和偏移。

所以 BN、LN 通常接：

$$
y_i=\gamma_i z_i+\beta_i.
$$

常见初始化是：

$$
\begin{aligned}
\gamma_i&=1, \\
\beta_i&=0.
\end{aligned}
$$

随后网络自己学习合适的缩放和平移。

这也解释了一个重要细节：

> **“均值为 0、方差为 1”描述的是归一化核心，不一定是最终输出。**

$$\gamma$$、$$\beta$$ 之后，最终输出不必保持这些统计量。

同时，$$\gamma$$、$$\beta$$ 并不能普遍恢复 LN 已经移除的、每个样本各不相同的原始均值；它们主要提供可学习的表达自由度。

## 九、放在一起比较

| 比较项                 | BatchNorm                      | LayerNorm                | RMSNorm            |
| ---------------------- | ------------------------------ | ------------------------ | ------------------ |
| 典型统计集合           | 同一特征，跨样本；CNN 还跨空间 | 同一样本或 token，跨特征 | 通常与 LN 相同     |
| 减均值                 | 是                             | 是                       | 否                 |
| 分母                   | 标准差                         | 标准差                   | 均方根             |
| 归一化核心的目标       | 均值 0、方差约 1               | 均值 0、方差约 1         | RMS 约 1           |
| 训练时依赖其他样本     | 是                             | 否                       | 否                 |
| 通常需要运行统计量     | 是                             | 否                       | 否                 |
| Norm 本身训练/推理算法 | 通常不同                       | 通常相同                 | 通常相同           |
| 标准可学习参数         | $$\gamma$$、$$\beta$$          | $$\gamma$$、$$\beta$$    | $$\gamma$$         |
| 常见场景               | CNN                            | Transformer、RNN 等      | 许多现代大语言模型 |

如果对应的特征或通道数是 $$D$$，标准仿射参数量通常分别为：

$$
\begin{aligned}
\operatorname{Params}(\mathrm{BN})&=2D, \\
\operatorname{Params}(\mathrm{LN})&=2D, \\
\operatorname{Params}(\mathrm{RMSNorm})&=D.
\end{aligned}
$$

不过，这些参数量相对于大模型的线性层通常很小。RMSNorm 的主要价值不是省下几个参数，而是：

> **用更简单的尺度控制，在许多模型中取得良好的效果，并减少中心化相关的计算。**

它通常计算更简单，但实际速度取决于融合算子、内存带宽、硬件和实现，不能直接把 Norm 的提速等同于整个模型的提速。

## 十、为什么 Transformer 常用 LN / RMSNorm？

### 1. 每个 token 可以独立归一化

Transformer 处理的是形状为 $$B\times T\times D$$ 的张量。

沿 $$D$$ 做 LN 或 RMSNorm：

- 不依赖其他样本。
- 不需要合并不同 token 的统计量。
- 训练与推理的归一化规则一致。
- 很自然地适用于自回归生成。

BatchNorm 并不是绝对不能用于序列模型，但如果把时间位置一起统计，就需要处理 padding、变长序列以及因果信息泄漏等问题，通常不如逐 token 的 LN/RMSNorm 自然。

### 2. Norm 的“种类”和“位置”是两件事

Transformer 中还会讨论 **Pre-Norm** 和 **Post-Norm**。

设 $$F$$ 是 Attention 或 FFN 子层：

#### Pre-Norm：先归一化，再进入子层

$$
\mathbf{h}'=\mathbf{h}+F\bigl(\operatorname{Norm}(\mathbf{h})\bigr).
$$

```text
h ────────────────────┐
                      + → h'
h → Norm → Attention ─┘
```

残差主路径保留了直接通路，通常有利于深层模型训练稳定。

#### Post-Norm：残差相加后再归一化

$$
\mathbf{h}'=\operatorname{Norm}\bigl(\mathbf{h}+F(\mathbf{h})\bigr).
$$

```text
h ────────────┐
              + → Norm → h'
h → Attention ┘
```

因此：

- **LN / RMSNorm**：回答“如何归一化”。
- **Pre-Norm / Post-Norm**：回答“放在哪里”。

不要把它们混为一谈。

### 3. 为什么很多大模型选择 RMSNorm？

在许多大语言模型中，控制隐藏表示的整体尺度已经很有价值，显式去除均值未必是必需的。

RMSNorm 因而提供了一种更简单的选择：

- 保留尺度归一化。
- 省去中心化操作。
- 在许多现代架构中有良好的训练效果。

但这不意味着 RMSNorm 在所有任务中都优于 LN；它们移除的信息不同，效果仍然依赖架构和任务。

**最后记住：BN 和 LN 主要差在统计维度；LN 和 RMSNorm 主要差在是否中心化，以及分母用标准差还是均方根。Norm 控制的是表示的统计尺度，不是把数值变成概率，也不是把分布变成正态分布。**
