---
layout:     post
title:      "R 统计学习（ISLR）-- Learning Notes (I)"
subtitle:   "subtitle"
date:       2019-01-10 12:00:00
author:     "Paradise"
header-img: "img/post-bg.jpg"
header-style: text
tags:
    - 数据分析
    - 统计分析
    - 机器学习
    - R
    - 笔记
---

> 教材介绍：<https://book.douban.com/subject/26430936/ >
>
> 相关资源：<http://faculty.marshall.usc.edu/gareth-james/ISL/ >


# 第一章 导论

- 1.1 统计学习概述
- 1.2 统计学习历史
- 1.3 关于这本书
- 1.4 本书适合的读者群
- 1.5 记号与简单矩阵代数
- 1.6 本书的内容安排
    - 第 2 章主要介绍统计学习的基本技术和概念，包含 K 最邻近算法
    - 第 3、4 章是经典的线性回归模型和分类模型（线性回归、logistic 回归、线性判别）
    - 第 5 章介绍交叉验证和自助法
    - 第 6 章提供了一类集经典与现代与一体的线性模型，这些方法是在标准线性回归基础上的改进（逐步变量选择、岭回归、主成分分析、偏最小二乘法和 lasso ）
    - 第 7 章介绍一类在一元输入问题中颇有成效的非线性方法，之后应用到多输入模型
    - 第 8 章介绍树类模型，包括挂袋法、提升法、随机森林
    - 第 9 章介绍支持向量机
    - 第 10 章介绍聚类，重点介绍主成分分析、K-means 聚类和系统聚类方法。
- 1.7 用于实验和习题的数据集
    - 网站：<http://www-bcf.usc.edu/~gareth/ISL/>
    - 数据集主要在 ISLR 包里，其中 Boston 在 MASS 包里，USArrests 在 R 基础模块中


# 第二章 统计学习

## 2.1 什么是统计学习

> 用 f 表示自变量和因变量的函数关系，统计学习就是估计 f 的一系列方法。

- 2.1.1 什么时候需要估计 f
    - 估计 f 的主要原因有两个：预测（prediction）和推断（inference）
    - 预测：求函数对应关系 f，一般情况下 f 视为“黑箱”，即不关心 f 的具体形式，只关心准确地预测。其精确性依赖于可约误差和不可约误差。
- 2.1.2 如何估计 f
    - 可分为参数方法和非参数方法
    - 参数方法：建立模型，使用训练数据集去拟合（fit）或训练（train）模型，获得 f 的参数，比如最小二乘法拟合参数。
    - 非参数方法：不需要对函数形式事先做明确的假设，估计函数在平滑处理后尽可能与更多的数据点接近。
- 2.1.3 预测精度和模型解释性的权衡
    - 预测越精确，模型越复杂，自变量越多，模型解释性越弱。
- 2.1.4 指导学习与无指导学习
    - 半指导学习：一部分观测点可以同时观测到预测变量和响应变量，一部分观测不到响应变量。
- 2.1.5 回归与分类问题
    - 将被解释变量为定量数据的问题称为回归分析问题，对应定性的为分类问题。

## 2.2 评价模型精度

- 2.2.1 拟合效果检验
    - 在回归中常用的标准为 MSE（均方误差），选择的模型应该使测试数据集的 MSE 尽可能小
    - 在分类问题中的标准为错误率（误差比率，误差观测点的占比）
    - 实践中可能缺乏测试集，代替方案是使训练均方误差（tarin MSE）尽可能小，缺点是可能出现过拟合（亦即光滑度过高）
    - 上问题的对应解决方案称作交叉验证
- 2.2.2 偏差-方差的权衡
    - 在数学上，训练 MSE 的期望可以表达为 y_hat 的方差、y_hat 偏差的平方和误差项 error 的方差三者之和。因此存在这个权衡。
    - 一般而言，使用光滑度更高的方法所得的模型方差会增加，偏差会减小。这两个量相对的变化率影响 MSE 的大小。
- 2.2.3 分类模型
    - 一个好的分类器应使错误率尽可能小
    - 贝叶斯分类器：可以在数学上证明测试错误率存在最小值。平均来说，可以设计一个简单的分类器将每个观测值分配到它最可能在的类别中。这种方法称作贝叶斯分类器
    - 贝叶斯边界：划分不同类别归纳的边界
    - 贝叶斯分类器将产生最低的测试错误率，称为贝叶斯错误率
    - K 最邻近方法（KNN）：使用贝叶斯方法需要知道给定 X 后 Y 的条件分布，在现实中很难获取。许多方法尝试在给定 X 后估计 Y 的条件分布，如 K 最邻近法
    - KNN 分类器原理：首先识别训练集的 K 个最靠近 x0 的点集，用 N0 表示。然后对每个类别 j 分别用 N0 中的点估计一个分值作为条件概率的估计。然后再运用贝叶斯规则将测试观测值 x0 分到概率最大的类中
    - 正如回归设置中，训练错误率和测试错误率没有必然关系。一般而言，当使用光滑度更高的方法（K 值更小，决策边界更曲折），训练错误率将减小，但是测试错误率不定。随着 1/K 的增大，测试错误率的变化近似 U 型

## 2.3 实验：R 语言简介

```R
# 查看环境中所有创建的对象
ls()
# 创建一个数值矩阵
x <- matrix(data = c(1,2,3,4), nrow = 2, ncol = 2)
x <- matrix(c(1,2,3,4), 2, 2, byrow = TRUE)
# 正态分布随机变量(标准正态分布)
x <- rnorm(100, mean = 0, sd=1)
y <- rnorm(100, mean = 50, sd=.1)
# 相关系数
cor(x, y)
# 方差和标准差
var(y)
sd(y)
sqrt(var(y))
# 希望代码每次运行产生一样的随机数组
set.seed(213)	# 参数为任意整数
mean(rnorm(100))

#绘图基础
x <- rnorm(100)
y <- rnorm(100)
plot(x, y, col='green',
     xlab = 'x-axis', ylab = 'y-axis', main = 'normal array plot')
# 保存图片
pdf('figure.pdf')
dev.off()

# 使用 contour() 函数产生一个等高线图
x <- seq(-pi, pi, length.out = 50)
y <- x
f <- outer(x, y, function(x,y) cos(y) / (1+x^2))
contour(x, y, f)
fa <- (f-t(f)) / 2
contour(x, y, f, nlevels = 45, add = T)
#使用image()函数生成热力图
image(x, y, fa)
persp(x, y, fa, theta = 30, phi = 30)
```

<img src="https://img-blog.csdnimg.cn/2020041901293448.jpg">

```R
# 索引数据
A <- matrix(1:16, 4, 4)
dim(A)
A[1, 4]
A[c(1,2), c(1,2)]
A[1:3, 1:3]
A[1:2, ]
A[-c(1,2), ]
A[-c(1,2), -c(3,4)]

# 载入数据
library(ISLR)
# 写入时默认编码为UTF-8
write.table(Auto, 'Auto.csv', sep = ',', quote = FALSE, row.names = FALSE)
Auto2 <- read.csv('Auto.csv')
# 打开可视化的交互式编辑器
fix(Auto)
# 剔除包含缺失值的行
Auto <- na.omit(Auto)
# 查看数据
dim(Auto)
names(Auto)

# 将内置数据储存到环境变量
attach(Auto)
# 将定量数据转换成定性数据
cylinders <- as.factor(cylinders)		# 这时多了一个levels属性
# 对定性数据可以直接plot产生箱形图
plot(cylinders, mpg, col='red', varwidth=TRUE, xlab='cylinders', ylab='mpg')
plot(cylinders, mpg, col='red', varwidth=TRUE, horizontal=TRUE)
# 绘制直方图
hist(mpg, col = sample(1:8))			# 共有8种颜色
hist(mpg, col = 3, breaks = 15)

#散点图矩阵
pairs(Auto)
#指定变量绘制散点图矩阵
pairs(~mpg+displacement+horsepower+weight+acceleration, Auto)

#紧随绘图之后使用identify函数进行交互式看图
plot(horsepower, mpg)
identify(horsepower, mpg, name)
#输出统计信息
summary(Auto)
```

<img src="https://img-blog.csdnimg.cn/20200419013209752.jpg">


# 第三章 线性回归

> 回顾线性回归模型的主要思路以及常用的最小二乘法

## 3.1 简单线性回归

- 3.1.1 估计系数
    - 残差平方和：`RSS = sum(residual^2)`
    - 最小二乘法估计参数：
        - `beta1_hat = sum((x-mean(x)) * (y-mean(y)) / sum((x-mean(x))^2)`
        - `beta2_hat = mean(y)- beta1_hat * mean(x)`
- 3.1.2 评估系数估计值的准确性
    - 复习计量和统计中讨论的无偏性、置信区间、假设检验以及 P 值等问题
- 3.1.3 评价模型的准确性
    - 使用参数：残差标准误 RSE 和拟合度 R^2

## 3.2 多元线性回归

- 3.2.1 估计回归系数
    - 同样使用最小二乘法，最小化 RSS
- 3.2.2 一些重要问题
    - 响应变量和预测变量之间是否有关系？-- F 检验
    - 如何选定重要变量？-- 通过 F 检验获得的 P 值选择
    - 如何衡量模型拟合度？-- 常用 RSE 和 R^2

## 3.3 回归模型中的其他注意事项

- 3.3.1 定性预测变量
- 3.3.2 线性模型的扩展
- 3.3.3 潜在的问题
    - 非线性的响应-预测关系
    - 误差项的自相关
    - 误差项异方差性
    - 离群点
    - 高杠杆点（类似离群点）
    - 共线性

## 3.4 营销策划（例子）

## 3.5 线性回归与K最邻近法的比较
- KNN 更灵活，但是缺少解释性，而且存在维数灾难

## 3.6 实验：线性回归

### 线性回归分析

```R
# 导入库
librarY(ISLR)
library(MASS)

# 在数据编辑窗口打开 data.frame
fix(Boston)
# 在编辑器窗口打开
view(Boston)
# 查看 filed 和 document
names(Boston)
?Boston

# 简单线性模型
mod <- lm(medv~lstat, data=Boston)
# 查看模型参数
print(mod)
names(mod)
summary(mod)
view(broom::glance(mod))

# 系数估计值的置信区间
confint(mod)
# 计算置信区间和预测区间
predict(mod, data.frame(lstat=(c(5,10,15))), interval = 'confidence')
predict(mod, data.frame(lstat=(c(5,10,15))), interval = 'prediction')

# 绘制残差散点图
plot(predict(mod), residuals(mod))
# 学生化残差
plot(predict(mod), rstudent(mod))
# 残差表明数据存在一定的非线性,计算杠杆统计量
plot(hatvalues(mod))
#具有最大杠杆统计量的值
which.max(hatvalues(mod))

# 模型诊断图
par(mfrow=c(2,2))
plot(mod)
```

<img src="https://img-blog.csdnimg.cn/20200419013338356.jpg">

```R
# 在线性回归中引入交互项
lm(medv~lstat*age, data = Boston)		# 相当于medv~lstat+age+lstat:age

# 对预测变量进行非线性变化
mod1 <- lm(medv~lstat, data = Boston)
mod2 <- lm(medv~lstat+I(lstat^2), data = Boston)
summary(mod1)
summary(mod2)

#可以看到加入二次项后P值接近于0
#进一步量化二次拟合在何种程度上优于线性拟合
===============================================================
> anova(mod1,mod2)
Analysis of Variance Table

Model 1: medv ~ lstat
Model 2: medv ~ lstat + I(lstat^2)
  Res.Df   RSS Df Sum of Sq     F    Pr(>F)    
1    504 19472                                 
2    503 15347  1    4125.1 135.2 < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
===============================================================

# 可视化二次模型，可以看到残差中可识别规律更少，即拟合效果更好 
par(mfrow=c(2, 2))
plot(mod2)
```

<img src="https://img-blog.csdnimg.cn/20200419013451466.jpg">

### 加入非线性变量

```R
# 创建更高阶多项式拟合
mod5 <- lm(medv~poly(lstat,5), data = Boston)	# 使用poly生成5阶多项式
summary(mod5)

# 进行对数变换
log_mod <- lm(medv~log(rm), data = Boston)
summary(log_mod) 

# 由于对数化后还是单调递增的，可以sort一下方便画成曲线
points(sort(Boston$rm), sort(predict(log_mod)), col='red', type='l', lwd=3)
bools <- residuals(log_mod) < 0
points(Boston$rm, residuals(log_mod), col='red', type = 'h')
points(Boston$rm[bools], residuals(log_mod)[bools], col='green', type = 'h')
```


# 第四章 分类

- 响应变量为定量变量的问题为回归，对应定性变量的问题的分类
- 应用最广泛的三种分类模型： 
    - 逻辑斯蒂回归（logistic regression）
    - 线性判别分析（linear discriminant）
    - K 最邻近（K-nearest neighbor）

## 4.1 分类问题概述

## 4.2 为何线性回归不可用

- 为什么不直接用二值化的虚拟变量代表响应变量？比如将某个三类分类问题经过编码给每一类分别赋值为 1、2、3？
- 这样做实际上默认了一个有序的输出，默认第 2 类到第 1、3 类的差距是一样的。因为有顺序性，不同的编码实际上会产生一个不同的回归模型。
- 注意到如果是两类变量可以直接使用二值化线性回归（哑变量方法）。同时，如果多类问题中，各类别确实有相应的顺序关系，那么也可以通过编码进行线性回归。

## 4.3 逻辑斯蒂回归

- 4.3.1 逻辑斯蒂模型
    - 模型基本形式：`p(X) = beta0 + beta1*X`，其中：`p(X) = P(Y=1|X)`
    - 存在的问题是拟合出来的 Y 预测值可能出现负值，因此使用逻辑斯蒂函数进行“归一化”：
        - `p(X) = e^(beta0 + beta1*X) / [1 - e^(beta0 + bata1*X)]`
    - 相关的术语：发生比（odd）、对数发生比（log-odd）、分对数（logit）
- 4.3.2 估计回归系数
    - 由于 beta 参数未知，因此需要训练数据对参数进行估计。可以使用最小二乘法进行估计，但是极大似然方法有更好的估计效果。
- 4.3.3 预测
- 4.3.4 多元 logistic 回归
    - 模型形式的扩展与多元线性模型类似
- 4.3.5 响应分类数大于 2 的 logistic 回归
    - 虽然二元的 logistic 回归可以扩展到多类问题，但是实践中不常用

## 4.4 线性判别分析（linear discriminant analysis, LDA）

- logistic 回归建立了直接估计 p(Y=k | X=x) 的模型，而线性判别分析则是分别对每种响应分类建立预测变量 X 的分布模型，再利用贝叶斯定理反过来去估计 p(Y|X)
- 当分布接近正态分布，线性判别分析模型在形式上与 logistic 模型相似
- 为何需要使用线性判别分析方法：
    - 当类别区分度较高时，logistic 模型估计的参数不够稳定；
    - 当样本量较小，而且每一类响应分类中对应得 X 接近正态分布时，线性判别分析模型更加稳定；
    - 应用到多类别分类时，使用线性判别分析模型。

- 4.4.1 运用贝叶斯定理进行分类
- 4.4.2 `p=1` 的线性判别分析
- 4.4.3 `p>1` 的线性判别分析
- 4.4.4 二次判别分析（quadratic discriminant analysis, QDA）

## 4.5 分类方法的比较

## 4.6 实验：logistic 回归、LDA、QDA、KNN

### **（1）股票市场数据**

```R
"	首先对 ISLR 库中的 Smarket 数据集进行统计分析。数据集包含 2001~2005 共 1250 
天里标普500指数的投资回报率。具体变量：
	Lag1 ~ Lag5：过去的 5 个交易日的投资回报率
	Volume：当天成交量
	Today：当天的投资回报率
	Direction：走势，Up 或 Down
"
library(ISLR)
dim(Smarket)
names(Smarket)
summary(Smarket)
pairs(Smarket)
attach(Smarket)

"------------------------ Logistic Regression ------------------------"

# 将数据分为训练集和测试集
train <- Smarket[(Year<2005), ]
test <- Smarket[!(Year<2005), ]

# 使用广义线性模型，其中包含了 logistic 模型
glm_mod <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,
              data = train,
              family = binomial)
summary(glm_mod)
# 模型的系数
coef(glm_mod)
summary(coef(glm_mod))

# 预测在给定输入情况下，市场上涨的概率
glm_mod_prob <- predict(glm_mod, test, type='response')
# 将概率转化为离散变量 Up 和 Down
glm_mod_pred <- rep('Down', 252)
glm_mod_pred[glm_mod_prob > 0.5] <- 'Up'

# 计算混淆矩阵
table(glm_mod_pred, test$Direction)
=====================================
> table(glm_mod_pred, test$Direction)
glm_mod_pred Down Up
        Down   77 97
        Up     34 44
=====================================

# 计算正确率
mean(glm_mod_pred == test$Direction)	# 0.4801587

# 优化模型：减少变量，选择 p 值较低的变量进行拟合。

"-------------------- Linear Discriminant Analysis --------------------"

library(MASS)

# 模型拟合
lda_mod <- lda(Direction~Lag1+Lag2, data=train)
plot(lda_mod)

# 计算预测值
lda_mod_pred <- predict(lda_mod, test)
names(lda_mod_pred)

# 计算混淆矩阵和正确率
table(lda_mod_pred$class, test$Direction)
mean(lda_mod_pred$class == test$Direction)

"------------------ Quadratic Discriminant Analysis ------------------"

qda_mod <- qda(Direction~Lag1+Lag2, data=train)
qda_mod_pred <- predict(qda_mod, test)
table(qda_mod_pred$class, test$Direction)
mean(qda_mod_pred$class == test$Direction)

"------------------------- K Nearest Neighbor -------------------------"

library(class)

# 准备数据
train.X <- cbind(Lag1, Lag2)[(Year<2005), ]
train.Y <- Direction[(Year<2005)]
test.X <- cbind(Lag1, Lag2)[!(Year<2005), ]

# 因为 KNN 算法具有随机性，需要 set.seed
set.seed(1)
knn.pred <- knn(train.X, test.X, train.Y, k=1)	# 更改 k 值寻找最佳拟合
table(knn.pred, test$Direction)
mean(knn.pred == test$Direction)
```

### **（2）Caravan 保险数据**

```R
attach(Caravan)

"	由于KNN分类模型的原理是通过圈定距离最近的观测来实现
	于是变量的尺度将对结果产生影响
	取值范围较大的变量比较小的变量对距离有更大的影响
	假设数据集包含 salary 和 age
	那么以美元和年来衡量的时候，与以日元和分钟来衡量的时候，会产生完全不同的分类结果
	解决办法：对数据进行标准化，使变量以0为均值，以1为标准差
"
# 标准化数据
standardized.X <- scale(Caravan[, -86])
var(standardized.X[, 1])
# 分成训练集和测试集
train.X <- standardized.X[-(1:1000), ]
test.X <- standardized.X[(1:1000), ]
train.Y <- Purchase[-(1:1000)]
test.Y <- Purchase[(1:1000)]

# KNN 模型
knn.pred <- knn(train.X, test.X, train.Y, k=5)
table(knn.pred, test.Y)
# 错误率
mean(test.Y != knn.pred)
mean(test.Y != 'No')

# logistic 模型
glm.fit <- glm(Purchase~., data=Caravan, family=binomial, subset=-(1:1000))
glm.probs <- predict(glm.fit, Caravan[(1:1000), ], type='response')
glm.pred <- rep('No', 1000)
glm.pred[glm.probs>0.5] <- 'Yes'
table(glm.pred, test.Y)
# 调整 logistic 模型预测的阈值
glm.pred<-rep('No',1000)
glm.pred[glm.probs>0.25]<-'Yes'
table(glm.pred,test.Y)
```


# 第五章 重抽样方法

- 通过反复从训练集中抽取样本，然后对每一个样本拟合一个感兴趣的模型，来获得关于拟合模型的附加信息。
- 重抽样方法可能产生计算量上的代价，因为需要反复地进行拟合
- 两种最常用的重抽样方法：
    - **交叉验证法（cross-validation）**
    - **自助法（bootstrap）**
- 交叉验证法可用来估计一种特定的统计学习方法的测试误差，来评价该方法的表现，或为该方法选择合适的光滑度。将上述过程分别称为模型评价和模型选择。
- 自助法应用范围很广，最常用于为一个参数估计或一个统计学习方法提供关于准确度的测量。

## 5.1 交叉验证法

    上一章讲到训练错误率和测试错误率之间存在较大的区别，这是验证训练模型时存在的主要问题。但是实际中经常缺少作为测试集的数据，对此有很多方法根据可获得的训练数据估计测试错误率，一些方法也使用数学方法对训练错误率进行修正。在本节中主要考虑保留训练数据的一个子集进行测试

- 5.1.1 验证集方法（validation set approach）
    - 原理：将观测集随机地分为两部分，一个训练集（training set）和一个验证集（validation set），或称为保留集（hold-out set）。
    - 缺陷：测试错误率波动较大（由于随机分割）；用于训练的观测数据量减少。
- 5.1.2 留一交叉验证法（leave-one-out cross-validation）
    - 原理：保留一个观测作为验证数据，其他作为训练数据。重复上述过程 n 次得到 n 个均方误差 MSE
    - 相对于验证集方法，LOOCV 方法的优缺点：
        - 优点：不容易高估测试错误率；减小了波动性。
        - 缺点：由于需要进行多次的拟合，计算量很大。
- 5.1.3 k 折交叉验证法（k-fold CV）
    - 原理：LOOCV 的一种替代，由重复拟合 n 次改为重复拟合 k 次
- 5.1.4 k 折交叉验证法的“偏差-方差”权衡
    - LOOCV 方法会产生一个近乎无偏的估计，但是方差较大。而选择适当的 k 值，k 折交叉验证法的方差会小很多。一般选取 k=5 或 k=10。
- 5.1.5 交叉验证法在分类问题上的应用
    - 在定量数据的交叉验证中，取 MSE 的均值；
    - 而对于定性问题，取错误率 Err 的均值。

## 5.2 自助法（bootstrap）

- 自助法是一种统计工具，用于衡量一个指定的统计量或统计学习中的不确定因素，如一个线性模型系数的标准差。

## 5.3 实验：交叉验证与自助法

```R
library(ISLR)
set.seed(1)
attach(Auto)

"------------------------------ 验证集方法 ------------------------------"

# 估计在 Auto 数据集上拟合多个线性模型产生的测试错误率
train <- sample(392, 196)
lm.fit <- lm(mpg~horsepower, data=Auto, subset=train)

# 均方差 MSE
mean((mpg-predict(lm.fit, Auto))[-train]^2)		# 22.28447

# 多项式模型
lm.fit2 <- lm(mpg~poly(horsepower, 2), data = Auto, subset = train)
lm.fit3 <- lm(mpg~poly(horsepower, 3), data = Auto, subset = train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)		# 16.4932
mean((mpg-predict(lm.fit3,Auto))[-train]^2)		# 16.4776

"---------------------------- 留一交叉验证法 ----------------------------"

library(boot)
# 使用 boot::cv.glm 对 glm 模型进行交叉验证
glm.fit <- glm(mpg~horsepower, data=Auto)
cv.err <- cv.glm(Auto, glm.fit)
# 交叉验证结果
cv.err$delta

"---------------------------- k 折交叉验证法 ----------------------------"

set.seed(17)
cv.error.10 <- rep(0, 10)
for (i in 1:10){
	glm.fit <- glm(mpg~poly(horsepower, i), data=Auto)
    cv.error.10[i] <- cv.glm(Auto, glm.fit, K=10)$delta[1]
}
plot(cv.error.10)

"------------------------------- 自助法 -------------------------------"

# 自助法几乎可以应用在所有情形，而不要求复杂的数学运算。
# 先创建一个计算统计量的函数，然后使用 boot 函数反复从数据集中有放回地抽取观测执行

# 创建一个计算感兴趣的统计量的函数
alpha.fn <- function(data, index){
	X <- data$X[index]
    Y <- data$Y[index]
    output <- (var(Y)-cov(X,Y)) / (var(X)+var(Y)-2*cov(X,Y))
    return (output)
}
# 计算统计量
alpha.fn(Portfolio, 1:100)
# 随机观测序列的统计量
alpha.fn(Portfolio, sample(100, 100, replace=TRUE))
# 自主法多次计算
boot(Portfolio, alpha.fn, R=100)
=====================================================
> boot(Portfolio, alpha.fn, R=100)

ORDINARY NONPARAMETRIC BOOTSTRAP

Call:
boot(data = Portfolio, statistic = alpha.fn, R = 100)

Bootstrap Statistics :
     original     bias    std. error
t1* 0.5758321 0.00278335  0.08577495
=====================================================
```


# 第六章 线性模型选择与正则化

- 标准的拟合线性模型的方法为最小二乘法，本章讨论其替代方法以及优化。
- 优化的目标是更高的预测准确率和更好地模型解释力
- 预测准确率：
    - 当相应变量与预测变量有近似线性关系，OLS 估计的偏差较小；
    - 当观测数量 n 远远大于参数个数 p，OLS 估计的方差通常较低；
    - 当 p>n，OLS 估计不唯一。
- 模型解释力：当模型存在非线性或者与模型无关的变量，OLS 通过系数为 0 提高解释力；本章将介绍几种自动进行特征选择和变量选择的方法。
- 主要的替代方法：子集选择、压缩估计、降维法

## 6.1 子集选择

- 6.1.1 最优子集选择
    - 原理：
        - 对 p 个预测变量的所有可能组合分别使用最小二乘法回归进行拟合
        - 对含有1个预测变量的模型，拟合 p 个模型；
        - 对含有两个预测变量的模型，拟合 p(p-1)/2 个模型
    - 以此类推。最后选择一个最优的模型。
- 6.1.2 逐步选择
- 6.1.3 选择最优模型

## 6.2 压缩估计方法

    上节的子集选择方法使用最小二乘法对包含预测变量子集的线性模型进行拟合。除此之外，还可以使用对系数进行约束或加惩罚的技巧。也就是将系数向 0 的方向压缩，以此提升拟合效果。常用的两种约束方法是 岭回归 和 lasso。

- 6.2.1 岭回归（ridge regression）
    - 原理：第三章介绍了最小二乘法，通过最小化 RSS 进行拟合。而岭回归通过最小化 `RSS + lambda*sum(beta^2)` 拟合。`lambda >= 0` 是调节参数，需单独确定。增加的项称为惩罚项，lambda 越大，压缩效果越明显，系数估计结果越趋于 0。
    - 注意到压缩的系数不应该包括 beta0，因为我们的目的是家所有缩减预测变量的系数，而不是缩减截距。因为截距用于测量当预测变量全为 0 时响应的均值。
    - l2范数：beta 系数平方求和再开方，衡量 beta 向量到原点的距离。在岭回归中，当lambda 变大，l2 范数变小。
    - 岭回归优点：
        - 与 OLS 相比，RR 衡量了误差和方差。
        - 随着 lambda 变大，RR 拟合的光滑度下降，方差降低，同时偏差变大。
        - 当最小二乘的方差较大时，RR 拟合可以有效降低方差。同时在计算量上也有微小优势。
- 6.2.2 lasso
    - 岭回归的劣势在于，子集选择方法会选择变量的一个子集进行建模，而 RR 的最终模型包含全部的 p 个变量。增加 lambda 只会减小系数绝对值，而不会剔除任何一个系数。当p的个数较大，不利于模型的解释。
    - lasso 原理：在 OLS 的 RSS 函数基础上改为：`RSS + lambda*sum(abs(beta))`
    - l1范数：beta 绝对值求和
    - 与岭回归类似，但是当 lambda 足够大，lasso 使用的 l1 惩罚项可以将某些系数强制设定为0。最后 lasso 回归只保留变量的一个子集，得到“稀疏模型”。原因是 l1 和 l2 范数的约束条件不同，在拟合中添加约束时，l1 范数对应的约束空间为菱形，l2为圆形。（详见教材）
    - 进一步对比：RR 和 lasso 并没有哪个是绝对好的。一般情况下，当一部分预测变量是真实有效的，而其他系数非常小的时候，lasso 比较出色。当这些系数都大致相等时，RR 比较出色。（然而这都是不能预先知道的）
- 6.2.3 选择调节参数
    - 需要调节的参数为：优化方程中的 lambda 和约束条件中的 s。一般使用交叉验证法选择最优参数。

## 6.3 降维方法

    上述方法的预测变量都来自原始的预测变量集，而降维法将预测变量进行转换，用转换后的变量拟合最小二乘模型。降维方法可以理解为寻找预测变量之间的相关性，主要方法有主成分分析和偏最小二乘。

- 6.3.1 主成分分析（principal components analysis, PCA）
    - 第十章将 PCA 作为无监督学习的一种工具进行更详细的讨论，这里将它作为回归降维方法进行介绍。
    - 在第一主成分上，数据的变化最大，定义了与数据最接近的那条线，使所有点到该线的垂直距离平方和最小。
    - 第二主成分 Z2 是所有与 Z1 无关的原始变量的线性组合中方差最大的。
    - 主成分回归方法（PCR）：是指构造前 M 个主成分 Z1~ZM，然后以这些主成分作为预测变量进行最小二乘拟合。
    - 特性上 PCR 和 lasso 十分类似，甚至可以认为 lasso 是连续型的 PCR。
    - 在 PCR 中，M 的值一般通过交叉验证来确定。
    - 进行 PCA 时，一般需要对变量进行标准化，使其度量在相同尺度上。否则方差较大的变量将在主成分中占主导地位。
- 6.3.2 偏最小二乘法（partial least squares, PLS）
    - 与 PCR 类似，作为降维的手段，取原始变量的线性组合作为新的变量集，然后进行最小二乘线性拟合。
    - 与 PCR 不同，PLS 通过有监督的方法进行新的特征的提取，也就是利用了相应变量的信息筛选新变量。这样不仅很好地近似了原始变量，同时与响应变量相关。简单来说，PLS 试图寻找一个可以同时解释预测变量和响应变量的方向。

## 6.4 高维问题

- 6.4.1 高维数据
    - 传统统计学方法有低维数据发展而来，但是现代的数据往往更复杂；特征数目
    - 比观测数目大的数据（p>n）称作高维数据。像 OLS 等传统方法已经不适于高维数据。
- 6.4.2 高维度下的问题
    - 包括“方差-偏差”权衡问题，过拟合问题等。
- 6.4.3 高维数据的回归
    - 前面提到的逐步选择、岭回归、lasso 以及 PCA 可以避免过拟合，可用于高维。
    - 维数灾难：特征数过大，引入了无关的特征，影响模型结果。
- 6.4.4 高维数据分析结果的解释

## 6.5 实验1：子集选择方法

```R
library(ISLR)
fix(Hitters)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))		# 存在缺失值
hitters <- na.omit(Hitters)		# 移除缺失值

"使用若干个与棒球运动员上一年比赛成绩相关的变量预测运动员的薪水"

# 使用 regsubset() 函数筛选最优模型，用法类似于 lm 函数
library(leaps)
regfit.full <- regsubsets(Salary~., hitters)
summary(regfit.full)$rsq

# 可视化模型结果
par(mfrow=c(1,2))
plot(summary(regfit.full)$rss,
     xlab='Number of Variables', ylab='RSS', type='l')
plot(summary(regfit.full)$adjr2, 
     xlab='Number of Variables', ylab='Adjusted RSq', type='l')
# 标记调整后的R方最大值
M <- which.max(summary(regfit.full)$adjr2)
points(M, summary(regfit.full)$adjr2[M], col='red', cex=2, pch=20)


"向前和向后逐步选择：在上例的基础上添加 method 参数决定 forward 还是 backward"

regfit.fwd <- regsubsets(Salary~., data=hitters, nvmax=19, method='forward')
regfit.bwd <- regsubsets(Salary~., data=hitters, nvmax=19, method='backward')
par(mfrow=c(1,3))
plot(coef(regfit.full, 7))
plot(coef(regfit.fwd, 7))
plot(coef(regfit.bwd, 7))
```

<img src="https://img-blog.csdnimg.cn/20200419013639244.jpg">

```R
"使用验证集方法和交叉验证法选择模型"

set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(hitters), replace=TRUE)

# 在训练集上进行最优子集选择
regfit.best <- regsubsets(Salary~., data=hitters[train, ], nvmax=19)
# 模型预测
test.mat <- model.matrix(Salary~., data=hitters[!train, ])
# 计算预测值和测试值的 MSE
val.errors <- rep(NA, 19)
for (i in 1:19){
	coefi <- coef(regfit.best, id=1)
    pred <- test.mat[, names(coefi)] %*% coefi
    val.errors[i] <- mean((hitters$Salary[!train])^2)
}
which.min(val.errors)
coef(regfit.best, 1)
```

## 6.6 实验2：lasso与岭回归

```R
library(ISLR)
Hitters <- na.omit(Hitters)
# 使用 glmnet::glmnet 函数，需要输入一个自变量矩阵和一个响应变量向量
x <- model.matrix(Salary~., Hitters)[, -1]
# model.matrix 函数不仅可以自动生成与预测变量相对应的矩阵
# 还能自动将定性变量转换为哑变量；注意到glmnet函数只能处理数值型输入
y <- Hitters$Salary

"------------------------------- 岭回归 -------------------------------"
library(glmnet)
grid <- 10^seq(10, -2, length=100)

# glmnet 的 alpha 参数确定模型，0 为岭回归，1 为 lasso
ridge.mod <- glmnet(x, y, alpha=0, lambda=grid)
# glmnet 函数默认将所有变量标准化，可以自定义 standardize=FALSE
dim(coef(ridge.mod))
# 使用l2范数，当 lambda 等于 11498 时的估计结果：
ridge.mod$lambda[50]
coef(ridge.mod)[, 50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))
# 使用 predict 函数获取新的 lambda 值对应的岭回归系数，如 lambda=50
predict(ridge.mod, s=50, type = 'coefficients')[1:20, ]

# 将数据分成训练集和测试集
set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
y_test <- y[(-train)]

# 基于训练集建立岭回归模型，并计算 lambda=4 时测试集的 MSE
ridge.mod <- glmnet(x[train, ], y[train], 
                    alpha=0, lambda=grid, thresh=1e-12)
ridge.pred <- predict(ridge.mod, s=4, newx=x[(-train), ])
MSE <- mean((ridge.pred-y_test)^2)		# 101036.8

# 使用交叉验证选择参数 lambda
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha=0)
plot(cv.out)
best.lambda <- cv.out$lambda.min
best.lambda		# 211.7416

# 使用最佳 lambda 时岭回归的 MSE
ridge.pred <- predict(ridge.mod, s=best.lambda, newx=x[(-train), ])
MSE <- mean((ridge.pred-y_test)^2)		# 96015.51
```

<img src="https://img-blog.csdnimg.cn/20200419013752553.jpg">



```R
"------------------------------- lasso -------------------------------"

# 使用同一个函数同一个数据集，操作与 ridge 类似
lasso.mod <- glmnet(x[train, ], y[train], alpha=1)
plot(lasso.mod)

# 使用交叉验证寻找最优参数
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha=1)
plot(cv.out)
best.lambda <- cv.out$lambda.min

# 计算最优参数对应的MSE
lasso.pred <- predict(lasso.mod, s=best.lambda, newx=x[(-train), ])
MSE <- mean((lasso.pred-y_test)^2)    # 结果MSE明显小于空模型和最小二乘模型

# lasso的稀疏性：结果模型中只含有7个变量，其他的系数为0
out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type='coefficients', s=best.lambda)[1:20]
sum(lasso.coef == 0)                      
```

<img src="https://img-blog.csdnimg.cn/2020041901383926.jpg">

## 6.7 实验3： PCR 与 PLS 回归

```R
library(ISLR)
Hitters <- na.omit(Hitters)

"------------------- principle component regression -------------------"

# 使用 pls 库中的 pcr 函数实现主成分回归
library(pls)
set.seed(2)
pcr.fit <- pcr(Salary~., data=Hitters, scale=TRUE, validation='CV')
summary(pcr.fit)

# 交叉验证得分（如MSE）图像：
validationplot(pcr.fit, val.type = 'MSEP')
validationplot(pcr.fit, val.type = 'R2')
# 结果当 M=16 时交叉验证误差最小
```

<img src="https://img-blog.csdnimg.cn/20200419013950189.jpg">



```R
# 下面使用最优子集选择方法
x <- model.matrix(Salary~., Hitters)[, -1]
y <- Hitters$Salary
set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

pcr.fit <- pcr(Salary~., data=Hitters, subset=train, scale=TRUE, validation='CV')
validationplot(pcr.fit, val.type='MSEP')
validationplot(pcr.fit, val.type='R2')
# 结果 M=7 时交叉验证误差较小

# 下面计算测试集误差
pcr.pred <- predict(pcr.fit, x[test, ], ncomp=7)
mean((pcr.pred-y.test)^2)
# 拟合结果较 RR 或 lasso 更好，但是由于 PCR 的建模机制，最终模型会更难解释

# 在最优参数上拟合PCR模型
pcr.fit <- pcr(y~x, scale=TRUE, ncomp=7)
=================================================================
> summary(pcr.fit)
Data: 	X dimension: 263 19 
	Y dimension: 263 1
Fit method: svdpc
Number of components considered: 7
TRAINING: % variance explained
   1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps
X    38.31    60.16    70.84    79.03    84.29    88.63    92.26
y    40.63    41.58    42.17    43.22    44.90    46.48    46.69
=================================================================

```

<img src="https://img-blog.csdnimg.cn/20200419014036613.jpg">

```R
"----------------------- Partial Least Squares -----------------------"

set.seed(1)
# 使用 plsr 函数，用法和 pcr 类似
pls.fit <- plsr(Salary~., data=Hitters, subset=train, scale=TRUE, validation='CV')
summary(pls.fit)
validationplot(pls.fit, val.type = 'MSEP')
validationplot(pls.fit, val.type = 'R2')

# 在最佳参数 M=2 上计算测试集 MSE
pls.pred <- predict(pls.fit, x[test, ], ncomp=2)
mean((pls.pred-y.test)^2)

# 最后使用 M=2 在整个数据集上建立 PLS 模型
pls.fit <- plsr(Salary~., data=Hitters, scale=TRUE, ncomp=2)
==================================
> summary(pls.fit)
Data: 	X dimension: 263 19 
	Y dimension: 263 1
Fit method: kernelpls
Number of components considered: 2
TRAINING: % variance explained
        1 comps  2 comps
X         38.08    51.03
Salary    43.05    46.40
==================================
```

<img src="https://img-blog.csdnimg.cn/20200419014129808.jpg">


# 第七章 非线性模型

- 线性模型更易于描述，实现简便，解释性能和推断理论较成熟。但是在预测上明显不足。需要在线性模型的基础上进行推广。
- 主要推广有：
    - 多线式回归（polynomial regression）
    - 阶梯函数（step function）
    - 回归样条（regression spline）
    - 光滑样条（smoothing spline）
    - 局部回归（local regression）
    - 广义可加模型(generalized additive model)

## 7.1 多项式回归

- 最简单的推广，就是将标准的线性模型换成一个多项式函数。

## 7.2 阶梯函数

- 在多项式回归中，使用特征变量的多项式函数作为预测变量，得到全局非线性的模型
- 如果希望得到局部非线性的模型，可以使用阶梯函数进行拟合。也就是将 X 的取值范围分成一些区间，每个区间拟合一个不同的常数。相当于将一个连续变量转换为一个有序的分类变量。

## 7.3 基函数

- 多项式和阶梯函数回归模型实际上是特殊的基函数方法。其基本原理就是对变量 X 的函数或函数的变换（b1(X), b2(X), ... ,bk(X)）进行建模。
- 常用的基函数转换除了上述两种，此外常见的还有小波变换和傅里叶级数等。

## 7.4 回归样条 -- 一类光滑的基函数

- 7.4.1 分段多项式
    - 分段多项式回归在 X 的不同取值区间拟合独立的低阶多项式函数，而不是在全局范围拟合高阶多项式函数。基本形式可以写为：
        - `y = beta0 + beta1*xi + beta2*xi^2 + beta3*xi^3 + ei`
    - 其中不同区域的 beta 系数的取值都不一样，系数变化的临界点称为结点（knot）
    - 多项式回归可视为无结点的分段多项式回归
- 7.4.2 约束条件与样条
    - 对于一个只有一个结点的三次多项式，共使用了 8 个自由度。
    - 为了使拟合曲线不过于光滑，并且结点之间没有跳跃，需要增加约束条件（即：连续性、一阶导数连续性、二阶导数连续性）。增加的每个约束都会有效地释放一个自由度，降低模型的复杂度。
    - 在一般情况下，与 K 个结点一同使用的三次样条会产生 4+K 个自由度。
- 7.4.3 样条基函数
- 7.4.4 确定结点的个数和位置
- 7.4.5 与多项式回归对比

## 7.5 光滑样条

- 7.5.1 光滑样条的简介
    - 上一节的回归样条方法：首先设定结点，然后产生一系列基函数，最后使用最小二乘法估计样条函数的系数。本节使用不同的方法。
- 7.5.2 选择光滑参数 lambda

## 7.6 局部回归

- 局部回归是拟合光滑非线性函数的另一种方法。在对一个观测 x0 进行拟合时，该方法只使用到这个点附近的观测。
- 变系数模型：局部回归的一种推广，一些变量进行全局回归，另外的进行局部回归。

## 7.7 广义可加模型

    GAM 提供了一种对标准线性模型进行推广的框架。在这个框架中，每一个变量用一个非线性函数进行替换，同时保持模型整体的可加性。

- 7.7.1 用于回归问题的 GAM
    - 前面的样条函数、局部回归、多项式回归或者之前章节的一些函数组合都可以用来产生 GAM。
    - GAM 的优点和不足：主要局限在于模型的形式被限定为可加形式，在多变量的情况下，通常会忽略掉有意义的交互项
- 7.7.2 用于分类问题的 GAM
    - 拟合给定预测变量值时，响应变量的值为 1 的概率，类似 logistic 回归。

## 7.8 实验：非线性建模



```R
library(ISLR)
attach(Wage)

"------------------------ 多项式回归和阶梯函数回归 ------------------------"

# 拟合 wage 的自由度为 4 的多项式模型
fit <- lm(wage~poly(age, 4), data = Wage)
coef(summary(fit))
# 更直接的方法
fit2 <- lm(wage~poly(age, 4, raw=TRUE), data = Wage)
coef(summary(fit2))
# 拟合fit2的等价形式
fit2a <- lm(wage~age+I(age^2)+I(age^3)+I(age^4), data = Wage)
fit2b <- lm(wage~cbind(age, age^2, age^3, age^4), data = Wage)

# 接下来构造一组age的值，进行预测和评估
agelims <- range(age)
age.grid <- seq(from=agelims[1], to=agelims[2])
preds <- predict(fit, newdata = list(age=age.grid), se=TRUE)
se.bands <- cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)

# 画出数据散点图以及用4次多项式函数拟合的结果
plot(age, wage, xlim = agelims, cex=0.5, col='darkgrey')
title('Degree-4 Polynomial')
lines(age.grid, preds$fit, lwd=2, col='blue')
matlines(age.grid, se.bands, lwd=1, col='blue', lty=3)
```

<img src="https://img-blog.csdnimg.cn/20200419014222607.jpg">

```R

"-------------------------------- 样条 --------------------------------"

# 使用 splines 包中的 bs 函数，默认生成3次样条
library(splines)
fit <- lm(wage~bs(age, knots = c(25, 40, 60)), data=Wage)
# 进行预测与评估
pred <- predict(fit, newdata = list(age=age.grid), se=TRUE)
plot(age, wage, col='gray')
lines(age.grid, pred$fit, lwd=2)
lines(age.grid, pred$fit + 2*pred$se.fit, lty='dashed')
lines(age.grid, pred$fit - 2*pred$se.fit, lty='dashed')

"--------------------- Generalized Additive Model ---------------------"

# 使用 education 以及 year 和 age 的自然样条函数作为预测变量拟合 GAM
gam <- lm(wage~ns(year,4)+ns(age,5)+education, data=Wage)

# 接下来使用 gam 包中的 s() 函数拟合光滑样条而不是自然样条
library(gam)
gam.m3 <- gam(wage~s(year,4)+s(age,5)+education, data=Wage)
# 画出结果
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE, col='blue')
# plot 能识别出 gam 对象，实际上是调用 plot.Gam() 函数进行绘图
plot.Gam(gam1, se=TRUE, col='red')
```

<img src="https://img-blog.csdnimg.cn/20200419014346799.jpg">


# 第八章 基于树的方法

- 本章介绍基于树的（tree-based）回归和分类方法，这些方法主要根据分层（stratifying）和分割（segmenting）的方式，将预测变量空间划分为一系列简单区域。对某个给定待预测的观测值，用它所属区域中的训练集的平均值或众数进行预测。

- 由于划分空间的分裂规则可以被概括为一棵树,又称作决策树（decision tree）方法。
- 基于树方法简便且易于解释，但准确率低于第六章和第七章的指导学习方法。将大量的树进行集成可以极大的提升预测准确性，但是会损失一些解释性。

## 8.1 决策树的基本原理

- 8.1.1 回归树（regression tree）
    - 最后划分出的区域称作树的终端结点（terminal node）或树叶（leaf）。决策树通常是由上往下画的，树叶位于树的底部。
    - 沿着树将预测变量空间分开的点称为内部结点（internal node），各个结点的连接部分称为分支(branch)。
    - 通过特征空间分层预测
    - 建立回归树的一般过程：
        - 将预测变量空间分割成 J 个不重叠的区域 R1~RJ，然后对落入区域 Rj 的每个观测值进行预测（取该区域上训练集的响应值的算术平均）。
    - 递归二叉分裂（recursive binary splitting）：一种自上而下的、贪心的方法。从顶部开始，每个结点分割两个分支。贪心（greedy）指在建立树的每一步中，最优的分裂只取决于当前的一步，而不考虑未来。
    - 树的剪枝：
        - 上述方法会有良好的预测结果，但是很可能造成过拟合，原因是这样产生的树可能过于复杂。
        - 裂点更少，区域数更少的树会有更小的方差和更好的解释性（以增加微小的偏差为代价）。一种策略是：仅当分裂使残差平方和的减小量达到某个阈值时，才分裂该结点。其缺点是：过于短视，因为一个结点很可能在以后的分裂中使 RSS 大度地减小。
        - 更好的策略是：形成一颗较复杂的树，再通过剪枝（prune）得到子树（subtree）。但由于子树的数量可能极其庞大，对对每一棵子树都进行交叉验证会太复杂。
        - 解决的方案有：代价复杂性剪枝（cost complexity pruning），也称作最弱联系剪枝（weakest link pruning）
- 8.1.2 分类树（classification tree）
    - 与回归树类似，区别在于分类树用来预测定性变量。构造与回归树也类似,使用了递归二叉树分裂。但是 RSS 指标替换为分类错误率
- 8.1.3 树与线性模型的比较
    - 当存在复杂的高度非线性，基于树方法将更优。
- 8.1.4 树的优缺点
    - 解释性强；更接近人的决策模式；可以用图形直观表示；可以直接处理定性的预测变量而不需要创建哑变量。缺点是预测不够准确。

## 8.2 装袋法、随机森林和提升法

- 8.2.1 装袋法（bagging）
    - 也称自助法聚集（bootstrap aggregation），类似于第五章的自助法。通过自助抽样创建初始训练集的多个副本，分别建立决策树，最后将这些树结合起来。
    - 袋外误差估计
    - 变量重要性的度量
- 8.2.2 随机森林（random forest）
    - 通过对树作去相关处理，实现了装袋法的改进。
- 8.2.3 提升法（boosting）
    - 同为装袋法的改进

## 8.3 实验：决策树

```R
library(ISLR)
library(tree)
library(MASS)
library(randomForest)
library(gbm)
```

### **（1）构建分类树 -- Carseats 数据集**

```R
attach(Carseats)

# 由于 Sales 为连续变量，需要记为二元变量
High <- ifelse(Sales <= 8, 'No', 'Yes')
# 加入原数据集
Carseats <- data.frame(Carseats, High)
# 分集
set.seed(1)
train <- sample(1:nrow(Carseats), 200)
Carseats.test <- Carseats[-train, ]
High.test <- High[-train]

# 使用除了 Sales 外的所有变量拟合 High 变量，tree() 的语法与 lm() 类似
tree.carseats <- tree(High~.-Sales, data=Carseats, subset=train)
tree.pred <- predict(tree.carseats, Carseats.test, type='class')
table(tree.pred, High.test)
plot(tree.carseats)
text(tree.carseats, pretty=0)

# 尝试进行减枝优化模型
set.seed(2)
# 交叉验证确定最优树
cv.caeseats <- cv.tree(tree.carseats, FUN=prune.misclass)
names(cv.carseats)    #分别给出叶子数、错误率、复杂性参数值、剪枝方法
# 画出错误率 dev 对 size 和 k 的函数
par(mfrow=c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type='b')
plot(cv.carseats$k, cv.carseats$dev, type='b')
# 树越大错误率越低，复杂度越高
```

### **（2）构建分类树 -- Boston 数据集**

```R
# 创建训练集并根据训练集生成树
set.seed(3)
train <- sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston <- tree(medv~., Boston, subset=train)
summary(tree.boston)
# 画出树
plot(tree.boston)
text(tree.boston, pretty=0)

# 交叉验证并生成剪枝树
cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type='b')
prune.boston <- prune.tree(tree.boston, best=5)
plot(prune.boston)
text(prune.boston, pretty=0)

# 用未剪枝的树对测试集进行预测
yhat <- predict(tree.boston, newdata=Boston[-train, ])
boston.test <- Boston[-train, 'medv']
abline(0, 1)
mean((yhat-boston.test)^2)
```

### **（3）装袋法与随机森林**

```R
# 装袋法：随机森林的一种特例
set.seed(4)
bag.boston <- randomForest(medv~., data=Boston, subset=train, mtry=13, 
                           importance=TRUE)
# mtry=13 意味着每个结点要考虑全部的 13 个变量。
# 使用测试集进行评估:
yhat.bag <- predict(bag.boston, newdata=Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag-boston.test)^2)		# 13.86875

# 使用ntree参数改变生成树的数量
bag.boston <- randomForest(medv~., data=Boston, subset=train, mtry=13,
                           ntree=25)
yhat.bag <- predict(bag.boston, newdata=Boston[-train, ])
mean((yhat.bag-boston.test)^2)		# 14.64803

# 生成随机森林与装袋法一样，只不过 mtry 小于变量数量，默认为 p/3
set.seed(5)
rf.boston <- randomForest(medv~., data=Boston, subset=train, mtry=6,
                          importance=TRUE)
yhat.rf <- predict(rf.boston, newdata=Boston[-train, ])
mean((yhat.rf-boston.test)^2)		# 13.72421
# 使用 importance 函数查看个变量的重要性,并绘图显示
importance(rf.boston)
varImpPlot(rf.boston)
```

<img src="https://img-blog.csdnimg.cn/20200419014500874.jpg">

### **（4）提升法**

```R
"	用 gbm 包对 Boston 数据集建立回归树
	由于是回归问题，gbm() 函数的 distribution 参数选用'gaussian'
	如果是二分类问题则选用 'bernoulli'
	对象 n.tree=5000 表示提升法模型共需要 5000 棵树
	选项 interaction.depth=4 限制每棵树的深度。
"
library(gbm)
set.seed(1)
boost.boston <- gbm(medv~., data=Boston[train, ], 
                    distribution='gaussian', 
                    n.trees=5000, 
                    interaction.depth=4)

# 使用summary生成相对影响图，并输出相对影响统计数据
summary(boost.boston)
# 画出最重要的两个变量的偏相关图（类似偏导数）
par(mfrow=c(2,1))
plot(boost.boston, i='rm')
plot(boost.boston, i='lstat')

# 使用测试集进行评估
yhat.boost <- predict(boost.boston, newdata=Boston[-train, ], 
                      n.trees=5000)
mean((yhat.boost-boston.test)^2)	# 16.50313
# 结果与随机森林接近

# 使用不用的压缩参数 lambda 进行提升法，默认值是 0.001，现在取 0.2
boost.boston <- gbm(medv~., data=Boston[train, ], 
                    distribution='gaussian', 
                    n.trees=5000, interaction.depth=4, shrinkage=0.2, 
                    verbose=FALSE)
# 测试均方差略低于默认压缩率
```

<img src="https://img-blog.csdnimg.cn/20200419014610914.jpg?">




# 第九章 支持向量机

> 支持向量机（support vector machine, SVM）是一种分类方法，在许多问题有较好的效果，是适应性最广的分类器之一。可以看作一类简单直观的最大间隔分类器（maximal margin classifier）。

## 9.1 最大间隔分类器

- 9.1.1 超平面（hyperplane）
    - 二维空间的超平面是一条直线，三维空间的超平面是一个平面。也就是说，在 p 维空间中，超平面是 p-1 维的平直的子空间。
    - 一个 p 维空间的超平面的表达式：
        - `beta0 + beta1*X1 + beta2*X2 + ... + beta_p*X_p = 0`
    - 使上式大于 0 的 X 落在超平面的一侧，使上式小于 0 的 X 落在另一侧
- 9.1.2 使用分割超平面分类
    - 根据训练数据以某种法则找出一个超平面，即一个线性决策边界。进行预测时，测试数据被分到哪一类取决于观测点落到超平面的哪一侧。
- 9.1.3 最大间隔分类器
    - 由于可以找到无数个超平面分割数据，因此需要一个选择标准。一种方法就是最大间隔分类器，也称最优分离超平面（optimal separating hyperplane）
    - 具体操作：
        - 首先计算训练集每个观测到一个特定的超平面的垂直距离，这些距离的最小值就是训练观测与分割超平面的距离，称作间隔（margin）
        - 最大间隔超平面就是使间隔最大化的一个分割超平面。
        - 距离分割超平面最近的观测点称作支持向量。
- 9.1.4 构建最大间隔分类器
- 9.1.5 线性不可分的情况
    - 在许多情况下并不存在分割超平面，因此需要支持向量分类器

## 9.2 支持向量分类器

- 9.2.1 概述
    - 即使存在分割超平面，上述分类器也经常不可取，因为对观测个体太敏感。
    - 支持向量分类器，也称软间隔分类器（soft margin classifier），允许小部分观测被错误分类，而保证分类器的稳定性。
- 9.2.2 细节
    - 落在间隔上或者落在错误的一侧的观测称为支持向量，因为只有这些观测会影响支持向量分类器。
    - 支持向量分类器只有一部分观测（支持向量）确定，意味着对于距离超平面较远的观测来说，分类器是鲁棒的。

## 9.3 狭义的支持向量机 -- 由线性决策边界过渡到非线性

- 9.3.1 使用非线性决策边界分类
    - 实际中会遇到非线性的分类。类似回归问题中使用多项式函数的非线性扩展，在支持向量分类器中可以类似地处理非线性问题。
    - 例如使用二次多项式或更高阶的多项式扩大特征空间，对于加入二次多项式，原来的 p 个特征将变成 2p 个特征，然后可以在更高维的空间中生成线性的决策边界。
- 9.3.2 支持向量机
    - SVM 是支持向量分类器的一种扩展，使用了核函数（kernel）来扩大特征空间。
    - 核函数的使用
- 9.3.3 心脏数据集的应用

## 9.4 多分类的SVM

- 扩展到 K 类主要有两种方法：一类对一类（One versus one），一类对余类。

## 9.5 与逻辑斯谛回归的关系

## 9.6 实验：支持向量机

```R
# e1071 包或者 LiblineaR 包都可以实现支持向量分类器和 SVM
library(ISLR)
library(e1071)
```

### **（1）支持向量分类器**

```R
# 首先生成属于两个类别的观测数据
set.seed(1)
x <- matrix(rnorm(20*2), ncol=2)
y <- c(rep(-1, 10), rep(1, 10))
x[y==1, ] <- x[y==1, ] + 1
# 检查是线性可分，结果非线性可分
plot(x, col=y+3)

# 构建数据集，将响应变量转换成因子变量
data <- data.frame(x=x, y=as.factor(y))
# 拟合支持向量分类器
svm.fit <- svm(y~., data=data, kernel='linear', cost=10, scale=FALSE)
# 可视化分类器
plot(svm.fit, data)
# 查看支持变量
svm.fit$index
# 使用更小的 cost 参数，可以得到更宽的间隔，更多的支持变量

# 使用 tune 函数进行交叉验证调参
set.seed(2)
tune.out <- tune(svm, y~., data=data, kernel='linear',
                ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100))
                )
plot(log10(tune.out$performances$cost), tune.out$performances$error, type='l', col='red')
points(log10(tune.out$performances$cost), tune.out$performances$dispersion, type='l', col='blue')
title(main='Error & dispersion with different cost')

====================================================================
> summary(tune.out$best.model)

Call:
best.tune(method = svm, train.x = y ~ ., data = data, 
          ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
          kernel = "linear")
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  linear 
       cost:  1 
Number of Support Vectors:  11
 ( 6 5 )
Number of Classes:  2 
Levels:  -1 1
====================================================================
```

<img src="https://img-blog.csdnimg.cn/2020041901470187.jpg">

### **（2）支持向量机**

```R
"	使用带核函数的支持向量机：
	kernel='polynomial -- 拟合多项式核函数的 SVM，调节 degree 参数；
	kernel='radial' -- 拟合径向基核函数的 SVM，调节 gamma 参数。
"
set.seed(3)

# 构建具有非线性决策边界的二维特征数据
x <- matrix(rnorm(200*2), ncol=2)
x[1:100, ] <- x[1:100, ] + 2
x[101:150, ] <- x[101:150, ] - 2
y <- c(rep(1, 150), rep(2, 50))
data <- data.frame(x=x, y=as.factor(y))
plot(x, col=y+5)

# 分集并拟合，使用径向基核函数，取 gamma=1
train <- sample(200, 100)
svm.fit <- svm(y~., data=data[train, ], kernel='radial', gamma=1, cost=1)
plot(svm.fit, data[train, ])

# 使用更大的 cost 减少误分观测，代价是边界不规则，可能会过拟合
svm.fit <- svm(y~., data=data[train, ], kernel='radial', gamma=1, cost=1e5)
plot(svm.fit, data[train, ])

# 使用 tune 进行交叉验证确定最佳的 gamma 和 cost 值
set.seed(4)
tune.out <- tune(svm, y~., data=data[train, ], kernel='radial',
                ranges=list(cost=c(0.1, 1, 10, 100, 1e3)),
                gamma=c(0.5, 1, 2, 3, 4))
library(tidyverse)
tune.out$performances %>% ggplot(aes(x=cost)) + 
	geom_line(aes(y=error), color='red') + 
	geom_line(aes(y=dispersion), color='blue') + 
	scale_x_log10() + 
	labs(y = 'error & dispersion',
		title = 'Error & dispersion with different cost')

===========================================
> summary(tune.out)
Parameter tuning of ‘svm’:
- sampling method: 10-fold cross validation 
- best parameters:
 cost
 1000
- best performance: 0.13
- Detailed performance results:
   cost error dispersion
1 1e-01  0.28 0.20439613
2 1e+00  0.15 0.10801234
3 1e+01  0.15 0.09718253
4 1e+02  0.15 0.10801234
5 1e+03  0.13 0.11595018
===========================================

# 使用最佳参数在测试集上进行评估
table(tune = dat[-train, 'y'], 
      pred = predict(tune.out$best.model, newx=data[-train, ]))

```

<img src="https://img-blog.csdnimg.cn/20200419014753425.jpg">

### **（3） ROC 曲线**（ Receiver Operating Characteristic）

```R
library(ROCR)

# 定义一个函数，在给定的包含每个观测值的 pred 和真实值下，画出ROC曲线
rocplot <- function(pred, truth, ...){
    predob <- prediction(pred, truth)
    perform <- performance(predob, 'tpr', 'fpr')
    plot(perform, ...)
}
# 拟合模型
svmfit.opt <- svm(y~., data=dat[train, ], 
                  kernel='radial', gamma=2, cost=1, decision.values=TRUE)
fitted <- attributes(predict(svmfit.opt, dat[train, ], 
                             decision.values=TRUE))$decision.values
# 绘制ROC曲线
rocplot(fitted, dat[train, 'y'], main='Training Data')
```

### **（4）多分类 SVM**

```R
"	使用khan数据集，数据集由 2308 个基因的表达测定组成；
	训练集和测试集分别由 63 和 20 个观测组成；"
names(Khan)
dim(Khan$xtrain)
dim(Khan$xtest)
table(Khan$ytrain)
table(Khan$ytest)

# 使用支持向量机预测癌症，由于特征数特别多，使用线性核函数
data <- data.frame(x=Khan$xtrain, y=as.factor(Khan$ytrain))
out <- svm(y~., data=data, kernel='linear', cost=10)
summary(out)
table(out$fitted, data$y)
# 可以看到训练集误差为 0，因为特征空间维度高，产生过拟合

# 测试集误差
data.test <- data.frame(x=Khan$xtest, y=as.factor(Khan$ytest))
pred.test <- predict(out, newdata=data.test)
table(pred.test, data.test$y)
```

## 第十章 无监督学习（unsupervised learning）

> 无监督学习是一系列统计工具，研究只有特征数据而没有响应变量的情况。其目标并非预测，而是寻找特征空间的有价值的模式。常见两种无监督学习：主成分分析（PCA）、聚类分析（clustering）。

## 10.1 无监督学习的挑战

    相比于前面介绍的有监督学习，无监督学习更具有挑战性。训练更倾向于主观性，不设定明确的分析目标。评价一个无监督学习的结果非常困难

## 10.2 主成分分析

    第六章研究主成分回归（PCR）时，解释了主成分方向：特征空间中原始数据高度变异（highly variable）的方向。获取主成分使用了 PCA 方法。

- 10.2.1 什么是主成分
  
    - 如果要将 p 个特征画出相关矩阵散点图，将有 p(p-1)/2 个散点图。而且每个图只包含数据集很小一部分信息，没有多大价值。使用 PCA 就可以得到一个二维表示来获得数据集的大部分信息。
    - 基本思想：n 个观测虽然都在 p 维空间中，但不是所有的维度同样地有价值。PCA 寻找尽可能少的有意义的维度（离散度越高越有价值）。通过PCA找到的每个维度都是原始的 p 个特征的线性组合。
    - 第一主成分是变量的标准化线性组合中方差最大的组合：
        - `Z1 = phi_11*X_1 + phi_21*X_2 + ... + phi_p1*X_p`
        - phi是希腊字母，代表载荷（loading）。标准化（normalized）是指为了防止载荷大小影响方差大小，限定载荷的平方和为 1。寻找第一主成分可以理解为在上述约束下最大化载荷平方和。
    
- 10.2.2 主成分的另一种解释
  
    <img src="https://img-blog.csdnimg.cn/20200419014853459.jpg">
    
    - 上图显示了一个三维数据集的前两个主成分载荷向量。这两个主成分张成一个平面，这个平面在空间中的方向是观测数据方差达到最大的方向。三维数据从主成分载荷方向投影到一个二维平面，得到一张二维图，即主成分得分向量。
    - 对此得到主成分分析的另一种解释：主成分提供了一个与观测数据最接近的低维线性空间（我理解为面积最小的投影）。
    
- 10.2.3 关于 PCA 的其他方面
  
    - 变量的标准化：在进行 PCA 之前，变量应该中心化使均值为零，还需要进行变量的标准化，某则影响结果。这是 PCA 与其他方法的重要区别。例如线性回归中，变量是否标准化对结果没有影响。
    - 主成分的唯一性：除了计算出来的符号正负不同（向量正方向不同），每个主成分的载荷向量是唯一的（即使用不同的软件包计算）。
    - 方差的解释比例（proportion of variance explained, PVE）：表示每个主成分解释了多少比例（百分比）的方差。计算公式略。
    - 决定主成分的数量：n*p 维的数据矩阵 X 有 min(n-1, p) 个不同的主成分，但是一般不会全部用上。用多少比较好，没有一个最佳答案。
    
- 10.2.4 主成分的其他用途

## 10.3 聚类分析方法

- clustering 是在一个数据集中寻找子群或类的方法，应用广泛。基本原则是，是每个类内的观测彼此尽量相似，不同类的差异尽量大。
- 聚类和 PCA 的目标都是用少量的概括性信息简化数据，但是机制不同：PCA 寻找观测的一个低维表示来解释大部分方差，聚类从观测中寻找同质子类。
- 主要聚类方法：K 均值聚类（K-means clustering）和系统聚类（hierarchical clustering）；前者预先规定类的数量，后者反之。

- 10.3.1  K 均值聚类
  
    - K-means 算法：
        - 1.确定 K 值，为每个观测随机初始化 1~K 的数值
        - 2.重复以下直到收敛：
            - a.分别计算 K 个类的中心，即该类中 p 维观测向量的均值向量
            - b.将每个观测分配到距离最近的类中心所属的类（欧氏距离的“最近”）
    - 由于 K-means 算法找到的是局部最优解，算法收敛是未必是总体最优解，还跟初始化有关。因此需要进行随机初始化，多次运行再对比找到最优解。
- 10.3.2 系统聚类法
  
    - 与 K-means 聚类对比，系统聚类法不需要预设类数 K，另外一个优点是可以输出一个好看的有关各观测的树型表示，称作谱系图（dendrogram）
    - 一种常见的系统聚类方法：自下而上（bottom-up）的方法，也称凝聚法（agglomerative）。其谱系图由叶子开始聚集到树干，形成一棵倒过来的树。
    - 解释谱系图：通过谱系图直观解释系统聚类法，略。
    - 算法实现：
        - 1. 计算 n 个观测中所有 n(n-1)/2 对观测数据之间的相异度（如欧氏距离），将每个观测看作一类。
        - 2. 令 i = n, n-1, n-2, ... ,2:
            - a.在 i 个类中，比较任意两个类的相异度，找到相异度最小的一对，并结合。用两个类之间的相异度表示两个类在谱系图中交汇的高度
            - b.计算剩余的 i-1 个类中，每两个类的相异度
        - LOOP
    - 四种常见的距离形式：最长距离法、最短距离法、类平均法、重心法。
    - 相异度指标的选择：一般都是欧氏距离
- 10.3.3 聚类分析的实践问题
    - 小策略撬动大理论：涉及很多方法或指标选择的问题
    - 验证聚类结果的问题：得到的是代表数据共性的子类，还是仅仅将噪声聚到了一起？可以通过 p 值进行显著性检验，但是没有公认的好方法。
    - 聚类分析的其他考虑：
        - 解释聚类分析结果的一个折中方法：建议对数据集的子集进行聚类分析，这样可以对所得到的类的稳定性有一个整体感知。

## 10.4 实验

### **（1）主成分分析**

```R
"使用 USArrests 数据集进行 PCA， 数据集包含50个州的观测，4个变量"
library(ISLR)
states <- row.names(USArrests)
variables <- names(USArrests)
# 查看变量的均值和方差，发现相差较大
apply(USArrests, 2, mean)
apply(USArrests, 2, var)

# 因此使用带有标准化的主成分分析
pr.out <- prcomp(USArrests, scale=TRUE)
pr.out$rotation		# 主成分载荷向量
pr.out$x			# 主成分得分向量
pr.out$sdev			# 主成分标准差

# 绘出前两个主成分的双标图
biplot(pr.out, scale=0)

# 计算每个主成分的方差解释比
pve <- pr.out$sdev^2 / sum(pr.out$sdev^2)
# 绘制每个主成分的 PVE 和累积 PVE
plot(cumsum(pve), 
     xlab='Principal Component', ylab='Cumulative PVE', 
     ylim=c(0,1), type='b', col='blue')
points(pve, ylim=c(0,1), type='h', col='green')
title(main='PVE & Cumulative PVE of Principal Component')
```

<img src="https://img-blog.csdnimg.cn/20200419014957507.jpg">

### **（2）聚类分析**

```R
"------------------------------- K Means -------------------------------"

# 创建数据
set.seed(1)
x <- matrix(rnorm(50*2), ncol=2)
x[1:25, 1] <- x[1:25, 1] + 3
x[1:25, 2] <- x[1:25, 2] - 3

# 使用较大的 nstart，可以模拟较多的随机初始化，并选取最佳结果
km.out <- kmeans(x, 2, nstart=20)
# 可视化模型结果
plot(x, col=(km.out$cluster+1), 
     main='K-means Clustering Results with K=2', 
     xlab='', ylab='', pch=20, cex=2)
# 注意到数据是二维的，容易可视化。对于多位数据，可以绘制前两个主成分的得分向量

# 对数据进行K=3的K均值聚类
set.seed(2)
km.out<-kmeans(x, 3, nstart=20)
plot(x, col=(km.out$cluster+1), 
     main='K-means Clustering Results with K=3', 
     xlab='', ylab='', pch=20, cex=2)
```

<img src="https://img-blog.csdnimg.cn/20200419015055933.jpg">

```R
"------------------------------ 系统聚类法 ------------------------------"

# 使用 hclust 函数进行分层聚类，使用不同的距离方式，以欧式距离作为相异度
hc.complete <- hclust(dist(x), method='complete')	# 最长距离法
hc.single <- hclust(dist(x), method='single')		# 最短距离法
hc.average <- hclust(dist(x), method='average')		# 类平均法

# 绘制谱系图，尽头的每个数字代表一个观测
par(mfrow=c(1,3))
plot(hc.complete, main='Complete Linkage', xlab='', sub='', cex=0.9)
plot(hc.single, main='Single Linkage', xlab='', sub='', cex=0.9)
plot(hc.average, main='Average Linkage', xlab='', sub='', cex=0.9)

# 使用 cutree 函数根据谱系图的切割获取各个观测的类标签
========================================================================
> cutree(hc.complete, 2)
 [1] 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
[35] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
========================================================================
```

<img src="https://img-blog.csdnimg.cn/20200419015146135.jpg">

```R
# 对标准化的数据进行系统聚类
x_sc <- scale(x)
plot(hclust(dist(xsc), method='complete'), 
     main='Hierarchical Clustering with Scaled Features',
     xlab='', sub='')

# 修改欧氏距离为基于相关性距离，使用 as.dist() 函数
x <- matrix(rnorm(30*3), ncol=3)
dd <- as.dist(1-cor(t(x)))
plot(hclust(dd, method='complete'), 
     main = 'Complete Linkage with Correlation-Based Distance', 
     xlab='', sub='')
```

<img src="https://img-blog.csdnimg.cn/20200419015342966.jpg">

### **（3）以 NCI60 数据集为例**

```R
# PCA 和系统聚类法常用于基因数据的分析
# NCI60 数据集由 64 个细胞的共 6830 个基因表达数据组成

nci.labs <- NCI60$labs		# 癌细胞类型
table(nci.labs)
nci.data <- NCI60$data		# 基因数据
dim(nci.data)

"--------------------- 使用主成分分析查找癌症相关细胞 ---------------------"

# 数据标准化的 PCA
pr.out <- prcomp(nci.data, scale=TRUE)
# 定义函数给每个观测对应的癌症类型分配不同颜色
Cols <- function(vec){
	cols <- rainbow(length(unique(vec)))
    return (cols[as.numeric(as.factor(vec))])
}
# 对前几个主成分进行可视化，绘制得分向量图
par(mfrow=c(1,2))
plot(pr.out$x[,1:2], col=Cols(nci.labs), pch=19, xlab='Z1', ylab='Z2')
plot(pr.out$x[,c(1,3)], col=Cols(nci.labs), pch=19, xlab='Z1', ylab='Z3')
# 可以看到对于同类癌症的细胞系的前几个主成分得分向量比较接近

# 绘制前几个主成分解释的方差
plot(pr.out)
# PVE 和累计 PVE
pve <- pr.out$sdev^2 / sum(pr.out$sdev^2)
plot(cumsum(pve), xlab='', ylab='', ylim=c(0,1), type='l', col='blue')
points(pve, xlab='', ylab='', ylim=c(0,1), type='h', col='green')
title(main='PVE & Cumsum-PVE')
```

<img src="https://img-blog.csdnimg.cn/20200419015427290.jpg">

```R
"---------------------- 对癌症的基因表现进行聚类分析 ----------------------"

sc.data <- scale(nci.data)

# 进行系统聚类并绘制谱系图
par(mfcol=c(1,3))
data.dist <- dist(sc.data)
plot(hclust(data.dist), labels=nci.labs, 
     main='Complete Linkage', xlab='', sub='', ylab='')
plot(hclust(data.dist, method='average'), labels=nci.labs, 
     main='Average Linkage', xlab='', sub='', ylab='')
plot(hclust(data.dist, method='single'), labels=nci.labs, 
     main='Single Linkage', xlab='', sub='', ylab='')
```

<img src="https://img-blog.csdnimg.cn/20200419015519904.jpg">

```R
# 在谱系图上某个高度切割可以产生指定类数的聚类，例如 4 类
hc.out <- hclust(dist(sc.data), method='complete')
hc.clusters <- cutree(hc.out, 4)
table(hc.clusters, nci.labs)

# 绘制这四个类的谱系图切割位置
par(mfrow=c(1,1))
plot(hc.out, labels=nci.labs, xlab='', sub='', ylab='')
abline(h=139, col='red')
```

<img src="https://img-blog.csdnimg.cn/20200419015622187.jpg">

```R
# 只对前几个主成分得分向量进行系统聚类，而不是全部变量
hc.out <- hclust(dist(pr.out$x[, 1:5]))
plot(hc.out, labels=nci.labs, 
     main='Hier Clust on First Five Score Vectors', 
     xlab='', sub='', ylab='')
table(cutree(hc.out, 4), nci.labs)
# 可将主成分获取的操作，看成是一种去噪处理。
```

<img src="https://img-blog.csdnimg.cn/20200419015659474.jpg">

--------------------------------------------

**END**

