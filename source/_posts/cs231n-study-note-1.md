---
title: cs231n study note 1
date: 2018-05-14 03:29:41
tags:
- machine learning
- cs231n
- knn
- loss function
- neural networks

categories: 
- 学习笔记
---


#一、学习内容

1. Data Driven Approach 数据驱动的方法
2. KNN K最近邻算法
3. Linear classifiers 线性分类器
4. Loss function 损失函数
5. Optimization 优化
6. Back propagation 反向传播
7. Neural Networks 神经网络


##1. 数据驱动
###1.1 思路
dataset: labels --> train --> predict
###1.2 CIFAR-10数据集
该数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。
![](cifar10.png)

- Data
```
一个10000*3072的numpy数组, 数据类型是无符号整形uint8. 这个数组的每一行存储了32*32大小的彩色图像(32*32*3通道=3072). 
按1024分组, 分别是red,green,blue通道. 图像是以行的顺序存储, 即前32个数是该图的像素矩阵的第一行.
```

- Labels
```
一个范围在0-9的含有10000个数的列表（一维的数组）。第i个数就是第i个图像的类标.
```




##2. K最近邻算法
- 分类器中最简单的例子
- 在dataset中寻找与input最相似的


###2.1 L1 distance / Manhatton distance
$$
d_1(I_1, I_2) = \sum_p|I_1^p - I_2^p|
$$

###2.2 复杂度分析
Train 		-- O(1)
Predict 		-- O(n)
`结论: 不理想 (VS CNN)`

###2.3 图像之间的视觉判断
`不同的图片, 使用KNN可以得到的结论是相同 PS某种情况下也是优势`
![](same.jpg)

###2.4 维度灾难
`随着维度的上升, 复杂度指数级增长`
![](knnCrash.jpg)


##3. 线性分类器
###3.1 参数模型中最简单的实例
###3.2 简单表达式:  
$$
f(x,W) = Wx + b
$$


###3.3 示例图
![CIFAR-10数据集实例](linearCla.jpg)

###3.4 线性分类器的劣势
![](hardTodo.jpg)

###3.4 两种解释方法
1. 每个种类的学习模板
2. 学习像素在高纬度空间的一个线性决策边界


##4. 损失函数
- dataset.  $ \langle x_i,y_i \rangle  _i^N = 1 $
- $ L = \frac{1}{N} \sum_iL_i(f(x_i,W),y_i) $
`目的是寻找一个合适的W, 使得L最小` find:  
$$ L_{min} $$

###4.1 multiclass SVN loss
$$ L_i = \sum_{j \neq y} max(0,s_j-s_{yj}+1) $$
1. $L_i  \in (0, +\infty)$
2. $ s\approx0 => L = c - 1 $
3. 如果求和中不排除相当的情况, 则L++
4. 使得L=0(L最小)的情况的W并不唯一, exp:2W
5. 如何在诸多使得L=0的W中选择最合适的
- 最简约
- 奥卡姆剃须刀
- Regularization: model should be simple so it works on test data
- 正则化
$$
L(W) = \frac{1}{N} \sum_{i=1}^N L_i (f(x_i,W),y_i) + \lambda R(W)
$$
- R(W)分类
1. L2  ==> $$ R(W) = \sum_k \sum_l W_{k,l}^2 $$
2. L1  ==> $$ R(W) = \sum_k \sum_l |W_{k,l}| $$
3. Elastic net (L1+L2)
4. Max norm
5. Dropout

###4.2 multinormial logistic regression
###多项逻辑斯蒂回归
###softmax loss

$$
P(Y=k|X=x_i) = \frac{e^{sk}}{\sum_j e^{sj}} where s=f(x_i;W)
$$

$$
L = -log P(Y=k|X=x_i)
$$
![](lossfun.jpg)


##5. 优化
###5.1 梯度下降 Gradient Descent
```
while true:
	weights_grad = evaluate_gradient(loss_fun, data, weights)
	weights += -step_size * weights_grad
```
step_size `步长` `学习率` `超参数` `优化第一步`


###5.2 Stochastic Gradient Descent(SGD)
$$
\nabla w L(W) = \frac{1}{N} \sum_{i=1}^N \nabla w L_i (f(x_i,W),y_i) + \lambda \nabla w R(W)
$$
`随机` `minibatch小批量` `蒙特卡洛估计`

1. 数值计算 -- 反向传播
2. 解析计算

##6. 反向传播
###6.1 导数/梯度过于复杂无法计算
###6.2 链式法则
###6.3 计算图
![](compuGraph.jpg)
###6.4 例证
$$ f(x,y,z) = (x+y)z $$
VS
$$ f(w,x) = \frac{1}{1+e^{-w_0x_0+w_1x_1+w_2}} $$
![](complex.jpg)
###6.5 向量-雅可比矩阵

##7. 神经网络初步
###多阶段分层计算
###多个W

![](cnn0.jpg)
![](cnn1.jpg)
###层数叠加 -- deep xxx
###activation function(类比神经元)
![](active.jpg)

###neural networks architectures
![](archt.jpg)


###code
```
class Neuron:
	def neuron_tick(inputs):
	cell_body_sum = np.sum(inputs * selt.weights) + self.bias
	firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))
	return firing_rate
```





#二、工作相关

![](bbs.jpg)
```
使用了python对每个帐号的封号进行机器学习
通过分析1个人的封号情况，分析所有人的封号情况
建立每个帐号自己的封号模型和总封号模型

如果被封了，那么这个时间段的封号数据模型就有了
接下来就是之前介绍的概率递归方案，所以，助手会越用越精准
所以不用反馈被封的问题，距离建模完成至少再封3次，距离精细模型完成至少还要7次
只有大家的模型基本上完成了，我们才能根据大家的封号模型来建立总的防封模型，才能get到被制裁的原理
```