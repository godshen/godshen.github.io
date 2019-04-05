---
title: 反向传播计算的作业
date: 2019-04-05 03:29:41
tags:
- deep learning
- neural networks
- back propagation

categories: 
- 学习笔记
---


##通用的三层前馈网络的BP学习算法的推导


### 输入矩阵

$$\boldsymbol{X}_{n \times1} =
\left[
 \begin{matrix}
   x_1 & x_2 & \cdots & x_n
  \end{matrix} 
\right]^T$$



### 从输入到隐藏层的参数矩阵

$$ \boldsymbol{W}_{m \times n}=
\left[
 \begin{matrix}
   w_{11} & w_{12} & \cdots & w_{1m}\\
   w_{21} & w_{22} & \cdots & w_{2m}\\
   {\vdots} & {\vdots} & {\ddots} & {\vdots} &\\
   w_{n1} & w_{n2} & \cdots & w_{nm}\\
  \end{matrix} 
\right]^T ​$$



### 隐藏层矩阵

$$\boldsymbol{H}_{m \times 1}
=\left[
 \begin{matrix}
   h_1 & h_2 & \cdots & h_m
  \end{matrix} 
\right]^T​$$



### 从隐藏层到输出的参数矩阵

$$\boldsymbol{V}_{l \times m}
=\left[
 \begin{matrix}
   v_{11} & v_{12} & \cdots & v_{1l}\\
   v_{21} & v_{22} & \cdots & v_{2l}\\
   {\vdots} & {\vdots} & {\ddots} & {\vdots} &\\
   v_{m1} & v_{m2} & \cdots & v_{ml}\\
  \end{matrix} 
\right]^T$$



### 计算输出

$$\boldsymbol{Y}_{l \times 1}
=\left[
 \begin{matrix}
   y_1 & y_2 & \cdots & y_l
  \end{matrix} 
\right]^T​$$



### 期望输出

$$\boldsymbol{D}_{l \times 1}
=\left[
 \begin{matrix}
   d_1 & d_2 & \cdots & d_l
  \end{matrix} 
\right]^T​$$




### 传递关系
$$
\boldsymbol{W} \cdot \boldsymbol{X} = \boldsymbol{H}
$$

$$
\boldsymbol{V} \cdot \boldsymbol{H} = \boldsymbol{Y}
$$

### 其展开后的求和公式形式为

$$
y_k = f(net_k) \quad k=1,2,...,l
$$
$$
net_k = \sum_{j=0}^m{v_{jk}h_j} \quad k=1,2,...,l
$$


$$
h_j = f(net_j) \quad j=1,2,...,m
$$
$$
net_j = \sum_{i=0}^n{w_{ij}x_i} \quad j=1,2,...,m
$$




### 单极性sigmoid函数

$$
f(x)={1 \over {1+e^{-x}}}
$$




### 误差计算
$$
E = {1 \over 2} (\boldsymbol{D}-\boldsymbol{Y})^2 = {1 \over 2} \sum_{k=1}^l(d_k-y_k)^2
$$




### 将以上误差定义式展开至隐藏层:
$$
E = {1 \over 2} \sum_{k=1}^l[d_k-f(net_k)]^2 = {1 \over 2} \sum_{k=1}^l[d_k-f(\sum_{j=0}^mv_{jk}h_j)]^2
$$



### 进一步展开至输入层:
$$
E = {1 \over 2} \sum_{k=1}^l\{d_k-f[\sum_{j=0}^mv_{jk}f(net_j)]\}^2 =  {1 \over 2} \sum_{k=1}^l\{d_k-f[\sum_{j=0}^mv_{jk}f(\sum_{i=0}^nw_{ij}x_i)]\}^2
$$



$$ \Delta v_{jk} = - \eta { \partial E \over \partial v_{jk}} \quad j=0,1,2,…m; k=1,2,..,l​$$

$$ \Delta w_{ij} = - \eta { \partial E \over \partial w_{ij}} \quad i=0,1,2,…n; j=1,2,..,m$$

### 式中负号表示梯度下降, 常数$\eta \in (0,1)$ 表示比例系数.



### 对于输出层, 上式可写成:

$$ \Delta v_{jk} = - \eta { \partial E \over {v_{jk}}} = - \eta { \partial E \over \partial net_k} \ { \partial net_k \over \partial v_{jk}} ​$$



### 对于隐藏层, 上式可写成:

$$ \Delta w_{ij} = - \eta { \partial E \over {w_{ij}}} = - \eta { \partial E \over \partial net_j} \ { \partial net_j \over \partial w_{ij}} ​$$



### 对于输出层和隐藏层各定义一个误差信号, 令
$$ \delta_k^y = - {\partial E \over \partial net_k } $$
$$ \delta_j^h = - {\partial E \over \partial net_j } $$

### 代入上式可得到

$$ \Delta v_{jk} = \eta \ \delta_k^y \ h_j$$

$$ \Delta w_{ij} = \eta \ \delta_j^h \ x_i$$



### 对于输出层, $\delta^y$可以展开为

$$\delta_k^y = - { \partial E \over \partial net_k} = - { \partial E \over \partial y_k} { \partial y_k \over \partial net_k} =  -  { \partial E \over \partial y_k} f'(net_k)​$$



### 对于隐藏层, $\delta^h$可以展开为

$$\delta_k^y = - { \partial E \over \partial net_j} = - { \partial E \over \partial h_j} { \partial h_j \over \partial net_j} =  -  { \partial E \over \partial h_j} f'(net_j)$$





### 对于输出层, 利用误差公式 $E = {1 \over 2} \sum_{k=1}^l(d_k-y_k)^2$, 可得
$$
{\partial E \over \partial y_k} = -(d_k - y_k)
$$

### 对于隐藏层, 利用误差公式 $E  = {1 \over 2} \sum_{k=1}^l[d_k-f(\sum_{j=0}^mv_{jk}h_j)]^2$, 可得
$$
{\partial E \over \partial y_k} = - \sum_{k=1}^l(d_k-y_k)f'(net_k)v_{jk}
$$

### 将Sigmoid激活函数代入, 可以得到



$$\delta_k^y = (d_k-y_k)y_k(1-y_k)​$$



$$\delta_j^h = \sum_{k=1}^l(d_k-y_k)f'(net_k)v_{jk}]f'(net_j) = (\sum_{k=1}^l \delta_k^y v_{jk})h_j(1-h_j)$$





### 所以得到三层前馈网络的BP学习算法权值调整计算公式为
$$
\Delta v_{jk} = \eta \delta_k^y h_j = \eta \ (d_k-y_k)y_k(1-y_k) \ h_j
$$

$$
\Delta w_{ij} = \eta \delta_j^h x_i = \eta \  (\sum_{k=1}^l \delta_k^y v_{jk})h_j(1-h_j) \  x_i
$$





##作业: 在此例中

### 输入层有4个, 隐藏层有3个, 输出有1个, 因此有

### n=4, m=3, l=1, 即为:

$$
\boldsymbol{X} =
\left[
 \begin{matrix}
   x_1 & x_2 & x_3 & x_4
  \end{matrix} 
\right]^T​
$$

$$ 
\boldsymbol{W}=
\left[
 \begin{matrix}
   w_{11} & w_{12} & w_{13}\\
   w_{21} & w_{22} & w_{23}\\
   w_{31} & w_{32} & w_{33}\\
   w_{41} & w_{42} & w_{43}\\
  \end{matrix} 
\right]^T ​
$$

$$
\boldsymbol{H}
=\left[
 \begin{matrix}
   h_1 & h_2 & h_3
  \end{matrix} 
\right]^T
$$

$$
\boldsymbol{V}
=\left[
 \begin{matrix}
   v_{11} \\ v_{21} \\ v_{31}
  \end{matrix} 
\right]^T ​
$$

$$
\boldsymbol{Y} = y​
$$

$$
\boldsymbol{D} = d​
$$

### 正向推导

$$
h_1 = w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + w_{41}x_4 = W_{t_1}X 
$$ 

$$
h_2 = w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + w_{42}x_4 = W_{t_2}X 
$$

$$
h_3 = w_{13}x_1 + w_{23}x_2 + w_{33}x_3 + w_{43}x_4 = W_{t_3}X ​
$$

$$
Y = y = v_{11}h_1 + v_{21}h_2 + v_{31}h_3 = V \ \left[
 \begin{matrix}
   W_{t_1}X \\ W_{t_2}X \\ W_{t_3}X
  \end{matrix} 
\right] = V \ W \ X​
$$

### 误差反向传播

$$
E = {1 \over 2} \sum_{k=1}^1\{d_k-f[\sum_{j=1}^mv_{jk}f(net_j)]\}^2 =  {1 \over 2} \sum_{k=1}^l\{d_k-f[\sum_{j=1}^mv_{jk}f(\sum_{i=1}^nw_{ij}x_i)]\}^2
$$

#### 代入n,m,l具体数值
$$
E = {1 \over 2} \sum_{k=1}^1\{d_k-f[\sum_{j=1}^3v_{jk}f(net_j)]\}^2 =  {1 \over 2} \sum_{k=1}^1\{d_k-f[\sum_{j=1}^3v_{jk}f(\sum_{i=1}^4w_{ij}x_i)]\}^2
$$

$$
E = {1 \over 2} \{d_1-f[\sum_{j=1}^3v_{j1}f(\sum_{i=1}^4w_{ij}x_i)]\}^2
$$

####误差为:

$$
\Delta v_{j1} = \eta \delta^y h_j = \eta \ (d_1-y_1)y_1(1-y_1) \ h_j
$$

$$
 \Delta v_{11} = \eta \ (d_1-y_1)y_1(1-y_1) \ h_1 ​
$$
$$
 \Delta v_{21} = \eta \ (d_1-y_1)y_1(1-y_1) \ h_2 
$$
$$
 \Delta v_{31} = \eta \ (d_1-y_1)y_1(1-y_1) \ h_3 
$$

$$
\Delta w_{ij} = \eta \delta_j^h x_i = \eta \  \delta^y v_{j1}h_j(1-h_j) \  x_i
$$

$$
 \Delta w_{11} =  \eta \  \delta^y v_{j1}h_1(1-h_1) \  x_1 
$$

$$
 \Delta w_{12} =  \eta \  \delta^y v_{j1}h_2(1-h_2) \  x_1 ​
$$

$$
 \Delta w_{13} =  \eta \  \delta^y v_{j1}h_3(1-h_3) \  x_1 
$$

$$
 \Delta w_{21} =  \eta \  \delta^y v_{j3}h_1(1-h_1) \  x_2 ​
$$
