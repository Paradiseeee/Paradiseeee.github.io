---
layout:     post
title:      "R for Data Science - Learning Notes (I)"
subtitle:   "R 数据分析"
date:       2018-11-22 12:00:00
author:     "Paradise"
header-img: "img/post-bg.jpg"
header-style: text
tags:
    - 数据分析
    - 编程语言
    - R
    - 笔记
---

> 教材介绍：<https://book.douban.com/subject/26757974/>
>
> 相关资源：<https://github.com/hadley/r4ds>


# CHPT01 ~ CHPT04

- Basic
- Scripts
- dplyr
- ggplot2

> [速查表](https://pan.baidu.com/s/1fiV7AQ_wiaCqcWnXKPYeUA ) \| 提取码：2ja3

# CHPT05 - Exploratory Data Analysis

**EDA 思路：**

- 带着问题进行迭代：
    + Visulising
    + Transforming
    + Modeling
    + New Question

**示例：在内置的 diamonds 数据集上进行数据探索。**

```R
# Q1.变量怎么变化？
diamonds %>% count(cut)                             # 离散变量频数
diamonds %>% count(cut_width(carat, 0.5))           # 连续变量频数

e <- ggplot(diamonds, aes(x=carat))                 # 定义绘图变量
e + geom_histogram(aes(color=cut), binwidth=0.1)    # 堆叠直方图
e + geom_freqpoly(aes(color=cut), binwidth=0.1)     # 分组密度图
e + geom_histogram(binwidth=0.01) +                 # 更精确的直方图
    coord_cartesian(ylim = c(0,100))                # 放大y轴

# 从分布图可以看到存在较多离群值
diamonds %>% filter(between(y, 3, 20))              # 过滤离群值，删除整行
diamonds %>% mutate(y=ifelse(y<3|y>20, NA, y))      # 用缺失值代替离群值
diamonds %>% mutate(bool=is.na(y))                  # 用新列标记缺失值
```

<img src="https://img-blog.csdnimg.cn/2020041501005035.jpg">

```R
# Q2.变量之间的相关性？

# Q2-1.连续变量与离散变量的相关性

# 使用分组密度图
e <- ggplot(diamonds, aes(x=price))
# cut 与 price 的相关性
e + geom_freqpoly(aes(color=cut), binwidth=500)
# 使用y=..density..显示标准化的分布，易于比较
e + geom_freqpoly(aes(y=..density.., color=cut), binwidth=500)

# 用箱形图
e <- ggplot(diamonds, aes(x=cut, y=price))
e + geom_boxplot() + coord_flip()
```

<img src="https://img-blog.csdnimg.cn/20200415010138856.jpg">

```R
# Q2-2.离散变量与离散变量的相关性

# 点阵图
ggplot(diamonds) + geom_count(aes(x=cut, y=color))
# 热力图
diamonds %>% count(color, cut) %>% ggplot(aes(x=color, y=cut)) + 
    geom_tile(aes(fill=n))
```

<img src="https://img-blog.csdnimg.cn/20200415010209142.jpg">

```R
# Q2-3.连续变量与连续变量的相关性

e <- ggplot(diamonds, aes(x=carat, y=price))
# 散点图（用灰度表示频数密度）
e + geom_point(alpha=3/100)

# 网格图
e + geom_bin2d()

# 将其中一个变量离散化
e + geom_boxplot(aes(group=cut_width(carat, 0.1)))
```

<img src="https://img-blog.csdnimg.cn/20200415010238413.jpg">

# CHPT06 - Workflow: Projects

- 不要依赖workspace保存环境，在代码里创造环境；
- 可以使用：`Ctrl+Shift+F10` 重启，然后用 `Ctrl+Shift+S` 重新运行 scripts 测试代码完整性；
- 把所有的输入数据，脚本，分析结果，图像关联到一个project里面；
- 使用相对路径而不是绝对路径。

# CHPT07 - Tibbles with tibble

## **数据处理的标准流程：**

- Import
- Tidy
- EDA (transforming --> visualising --> modeling)
- Communicate(report, visualising for communication)

**Tibble：**data.frame 的优化版本，功能更强

```R
# 创建 tibble 对象
tibble(x=1:5, y=1, z=x^2+y)
# 使用 tribble 函数创建
tribble(
    ~x, ~y, ~z,
    "A", 1, TRUE,
    "B", 2, FALSE
)
# 强制转换 data.frame 对象
as.tibble(iris)

# tibble 显示更整齐，显示数据类型，并且可以强制显示所有变量
nycflights13::flights %>% print(n=20, width=Inf)
# 索引方式
tb <- tibble(x=runif(10), y=rnorm(10))
tb$x; tb[["x"]]; tb[[1]]; tb $>$ .$x
```

# CHPT08 - Import Data with readr

## 基本操作

```R
data <- read.csv('C:/Users/Paradise/Desktop/test.csv')
data <- read_csv(
    "a, b, c
    1, 2, 3
    4, 5, 6", col_names=c('x', 'y', 'z')
)
# 使用 read_csv_chunked(file, chunk_size) 分块读取大文件
```

## 数据整理

```R
# 统一数据类型
lst = c('1, 2, data$b, TRUE, TRUE, FALSE')
parse_integer(lst)
parse_logical(lst)

# 字符串 --> ASCII 编码
charToRaw("AaBbCc")
# 编码 --> 字符串
eco <- '\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd'
parse_character(eco, locale=locale(encoding='Shift-JIS'))
# 匹配字符串编码方式
guess_encoding(charToRaw("こんにちは"))

# 时间日期
parse_datetime('2018-10-01T123456')
parse_time('12:34 am')
parse_date("10/01/18","%m/%d/%y")
parse_date("1 October 2018","%d %B %Y")
```

# CHPT09 - Tidy Data with tidyr

## 常见问题一：一个变量分布在多列

```R
exmp1 <- read_csv(
    "country,   2017,   2018
    Asgard,     745,    2666
    Wakanda,    7737,   8048
    Titan,      2125,   2376"
)
# 使用 gather 函数整理：
tidy1 <- exmp1 %>% gather('2017', '2018', key='year', value='values')

--------------------------------
> tidy1
# A tibble: 6 x 3
  country year  values
  <chr>   <chr>  <dbl>
1 Asgard  2017     745
2 Wakanda 2017    7737
3 Titan   2017    2125
4 Asgard  2018    2666
5 Wakanda 2018    8048
6 Titan   2018    2376
```

## 常见问题二：一个观测分布在多行

```R
exmp2 <- read_csv(
    "coutry,    year,   type,   values
    Asgard,     1999,   A,      745
    Asgard,     1999,   B,      18707
    Asgard,     2000,   A,      2666
    Asgard,     2000,   B,      59536
    Wakanda,    1999,   A,      7737
    Wakanda,    1999,   B,      30636
    Wakanda,    2000,   A,      8048
    Wakanda,    2000,   B,      50489"
)
# 使用 spread 函数整理（gather 的逆操作）
tidy2 <- exmp2 %>% spread(key='type', value='values')

---------------------------------
> tidy2
# A tibble: 4 x 4
  coutry   year     A     B
  <chr>   <dbl> <dbl> <dbl>
1 Asgard   1999   745 18707
2 Asgard   2000  2666 59536
3 Wakanda  1999  7737 30636
4 Wakanda  2000  8048 50489
```

## 常见问题三：一个列包含多个变量

```R
exmp3 <- read_csv(
    "country,   year,   heros
    Asgard,     1999,   Thor/Loki
    Asgard,     2000,   Odin/Heimdall
    Midgard,    1999,   Hulk/BlackWidow"
)
# 使用 separate 分离两个变量（逆操作：unite）
tidy3 <- exmp3 %>% separate(heros, into=c('A', 'B'))

---------------------------------
> tidy3
# A tibble: 3 x 4
  country  year A     B         
  <chr>   <dbl> <chr> <chr>     
1 Asgard   1999 Thor  Loki      
2 Asgard   2000 Odin  Heimdall  
3 Midgard  1999 Hulk  BlackWidow
```

# CHPT10 - Relational Data with dplyr

```R
# 确定变量取值是否唯一
library(nycfights13)
planes %>% count(tailnum) %>% filter(n>1)

# 联结表
filghts %>% left_join(airlines, by='carrier')
# 使用 mutate 实现同等效果：
mutate(flights, name=airlines$name[match(carrier, airlines$carrier)])

# Inner/Outer Joins：
x<-tribble(
    ~key,   ~x,
    1,      "x1",
    2,      "x2",
    3,      "x3"
)
y<-tribble(
    ~key,   ~y,
    1,      "y1",
    2,      "y2",
    4,      "y4"
)
inner_join(x,y,by="key")    # 结果值保留key相同的部分（交集）
left_join(x,y,by="key")     # 第一个参数的key保全，第二个参数进行匹配
right_join(x,y,by="key")    # 与上一例相反
full_join(x,y,by="key")     # 全部key的值都保留

# 使用 merge 实现以上对应的操作
merge(x, y)
merge(x, y, all.x=T)
merge(x, y, all.y=T)
merge(x, y, all.x=T, all.y=T)

# Filtering Joins
top_dest <- filghts %>% count(dest, sort=T) %>% head(10)
flights %>% filter(dest %in% top_dest$dest)     # top 10 目的地
# 使用 semi_join 代替手动进行 filter
semi_join(flights, top_dest)
# 使用 anti_join 找出不匹配的部分
anti_join(flights, planes, by='tailnum')

# 集合操作
intersect(x, y)     # 交集
union(x, y)         # 与非
setdiff(x, y)       # x - y
```

# CHPT11 - Strings with stringr

```R
# 字符串操作
library(stringr)

# str_ 函数族
str_length('ABC')
str_c('(', c('b', 'c', 'd'), ')', sep='-')
# [1] "(-b-)" "(-c-)" "(-d-)"           # 支持向量化
str_c(if (TRUE) 'FUCK!')                # 支持条件语句
str_sub('abcde', 1, 3)                  # 取子集：1-3
str_to_lower('ABC')                     # 转为小写
str_view('abcd', 'a.*?')                # 使用正则表达式匹配
str_detect(c('ab', 'bc', 'cd'), '.b')   # 同上，返回布尔型
str_subset(c('ab', 'bc', 'cd'), 'b.')   # 同上，返回子集
str_subset(c('ab', 'bc', 'cd'), 'a')    # 同上，返回计数值
str_view_all(c('ab', 'bc', 'cd'), 'ab') # 同上，全字匹配
str_split('a,b,c,d,e', ',')             # 按特定字符切割

# str_replace
# 替换第一个找到的匹配
c('abc', '123') %>% str_replace('[bc23]', '-')
# 替换全部匹配值
c('abc', '123') %>% str_replace_all('[bc23]', '-')
# 替换为不同的值
c('abc', '123') %>% str_replace_all(c('a'='-', '1'='+'))
```

# CHPT12 - Factors with forcats

```R
# 处理因子变量
library(forcats)

# 使用 factor 对象：
# 直接使用 sort 按字母排序
x <- c("Dec", "Apr", "Jan", "Mar")
sort(x) 
# 创建 levels
month_level <- c("Jan","Fed","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec")
# 将 x 转化为带有 level 的离散变量
y <- factor(x, levels=month_level)
# 对 factor 对象使用 sort 按照 level 排序
sort(y)
```

# CHPT13 - Dates and Times with lubridate

```R
# 使用 lubridate 处理时间日期
library(lubridate)
library(nycflights13)

# 常用函数
today()
ymd('2018-10-19')
mdy('January 31st, 2018')
dmy('31-Jan-2018')
mdy_h('10191812')           # 2018-10-19 12:00:00 UTC
ymd_hms('181019 120000')
ymd(20181019, tz='UTC')     # 支持数值型，支持指定时区

# 操作示例
flights %>% select(year, month, day, hour, minute) %>%
    mutate(departure=make_datetime(year, month, day, hour, minute))

# 其他创建时间日期对象的形式
as_datetime(today())
as_date(now())
exmp <- ymd_hms('20181019 120001')
year(exmp)
wday(exmp, label=TRUE, abbr=FALSE)

# 修改时间日期对象
year(exmp) <- 2019
hour(exmp) <- hour(exmp) + 1
update(exmp, year=2019)

# 时间跨度对象 durations （秒格式）
age <- today() - ymd(19950825)
age <- as.duration(age)
aweek <- ddays(7)

# 时间跨度对象 Periods （年月日时分秒格式）
years(1)+months(2)+days(3)+hours(4)+minutes(5)+seconds(6) 
# 用于计算时间推移
today() + years(1)

# 时区与本地化
Sys.timezone()      # 查看系统的时区
OlsonNames()        # 查看 R 支持的所有时区
t1 <- ymd_hms("20181020 000000", tz="America/New_York")
t2 <- ymd_hms("20181020 060000", tz="Europe/Copenhagen")
t1 - t2 == 0        #TRUE
```

# CHPT14~15 Pipes & Functions (Omitted)

# CHPT16 - Vectors

## 基础知识

- **两类向量：**
    - 原子型向量：
        - 整型、双整型、字符型、复数型、逻辑型、raw
    - 列表：循环向量、列表中包含列表
- NULL 与 NA：前者表示缺失的向量，后者表示确实的某个值
- 向量的两个关键性质：类型、长度
- 属性与增广矩阵：
    - 整型向量的因子
    - 数值型向量的时间日期
    - 列表的数据框

## 示例

```R
# （1）数值型 ----------------------------------------
typeof(1)       # double
class(1)        # numeric
typeof(1L)      # integer
class(1L)       # integer

# 整型（integer）和双整型（double）的区别：
# 所有双整型数据都是有限精度的浮点数
sqrt(2)^2 == 2              # FALSE
dplyr::near(sqrt(2)^2, 2)   # TRUE
# 双整型除了 NA 还有三类特殊值
c(-1, 0, 1) / 0             # -Inf NaN Inf
# 对应的判别函数：is.infinite() is.finite() is.nan()

# （2）字符型 ----------------------------------------
#重要特点：在R中字符型储存在全局环境，只储存一次，以减少空间占用
x <- "This is a really long string."
object.size(x)              #136 bytes
y <- rep(x, 1000)           #产生一个字符型向量
object.size(y)              #8128 bytes--一个指针8bytes，因此约为8000bytes

# （3）向量的强制转换 -----------------------------------
# 显式的强制转换：as.integer()、as.character()、as.double()、as.logical()
# 隐式的强制转换：
exmp <- 1:10
sum(exmp > 5)                   # logical --> integer
if (lenth(exmp)) {print('integer --> logical')}

# 强制转换的优先级
typeof(c(TRUE, 1L))             # integer
typeof(c(1L, 1.1))              # double
typeof(c(TRUE, 1L, 1.1, 'A'))   # character

# （4）向量化循环规则 ------------------------------------
1:10 + 1:3      # 2  4  6  5  7  9  8 10 12 11

# （5）向量取子集 ---------------------------------------
x <- c("one", "two", "three", "four", "five")
x[c(1, 3, 5)]
x[c(1, 1, 3, 3)]
x[c(-1, -3, -5)]        # 负值表示取其余集
x <- set_names(1:5,x)   # 命名
x["one"]                # 类型为 Named int，可以用名称进行索引

# （6）循环向量：列表 -------------------------------------
#列表可以包括列表，因此可以创建分层结构或树形结构
x_named <- list(a=c(1,2,3), b=c(4,5,6), c=c(7,8,9))
str(x_named)            #使用str函数打印列表
y <- list("a", 1.5, 1L, TRUE)
==============
> str(y)
List of 4
 $ : chr "a"
 $ : num 1.5
 $ : int 1
 $ : logi TRUE
==============

# （7）列表取子集 -----------------------------------------
exmp <- list(a=1:3, b="a string", c=pi, d=list(1,2))
exmp[1:2]
str(exmp[3])        # 返回向量
str(exmp[[3]])      # 返回数值
x_named[[3]]        # 返回数值向量7，8，9
x_named[[3]][1]     # 返回数值7
exmp$d              # 名称索引，等同于exmp[["d"]]

# （8）Attributes属性 ------------------------------------
x <- 1:10
attr(x, "greeting")
attr(x, "greeting") <- "hi"
attr(x, "farewell") <- "bye"
===============
> attributes(x)
$greeting
[1] "hi"
$farewell
[1] "bye"
===============
# 三种重要属性：names、dimensions、class；用于面相对象的S3系统
# S3系统：控制泛函的工作
# 泛函：输入的类不同函数的功能不同

# （9）增广向量 Augmented Vectors：带有附加属性的原子型向量 ----------
x <- factor(c("a","b","c"), levels=c("a","b","c","d"))
typeof(x)           # "interge"
===================
> attributes(x)
$`levels`
[1] "a" "b" "c" "d"
$class
[1] "factor"
===================
```

# CHPT17 - 使用 purrr 进行迭代

```R
library(tidyverse)
exmp <- tibble(a=rnorm(10), b=rnorm(10), c=rnorm(10), d=rnorm(10))

# 使用 for 循环计算每一行的中位数
output <- vector('double', ncol(exmp))
for (i in seq_along(exmp)) {
    output[i] <- median(exmp[[i]])
}

# 使用 map 代替 for 循环 ------------------------------------------------
output <- exmp %>% map(median)      # map 每一行
exmp %>% t(.) %>% map(median)       # map 每一列

# 在 map 中定义函数
# 按气缸数目cyl分成几部分，对每部分进行 mpg~wt 的线性回归
models <- mtcars %>% split(.$cyl) %>% map(
    function(exmp) lm(mpg~wt,data=exmp)
)

# 可以在map函数里面加入被引用函数的参数
mul <- list(3,7,-2)
mul %>% map(rnorm, n=5) %>% str()
# 如果还需要增加方差变量
sigma <- list(1,5,10)
seq_along(mul) %>% map(~rnorm(5, mul[[.]], sigma[[.]])) %>% str()

# 多个输入变量的map：map2()/pmap() ---------------------------------------

# 使用map2可以直接将map应用到两个向量，互不干扰
map2(mul, sigma, rnorm, n=5) %>% str()
# map2 算法：
map2inside<-function(x,y,f,...){        # 用省略号定义未知参数
  out<-vector("list",length(x))         # x与y的长度应相等
  for (i in seq_along(x)) {
    out[[i]]<-f(x[[i]],y[[i]],...)      # 向量化的操作
  }
  return(out)
}

# 当需要应用到多于两个的向量，可以使用pmap()
n<-list(1,3,5)
arg <- list(n,mul,sigma)
arg %>% pmap(rnorm) %>% str()
# 注意参数列表需要命名，并按照所引用函数的参数输入顺序
arg2 <- tribble(
    ~n,	~sd,~mean,
    1,	1,	3,
    3,	5,	7,
    5,	10,	-2
)
pmap(arg2, rnorm) %>% str()
#可以以数据框输入参数，参数不一定按顺序，但是要按照被引用函数的参数名称对参数变量进行命名

# 其他形式的for循环 ------------------------------------------------

# 谓语函数
keep(iris, is.factor) %>% str()     # 保留谓语返回TRUE时的部分，谓语指is.factor语句
discard(iris, is.factor) %>% str()  # 保留谓语返回FALSE时的部分
x <- list(1:5, letters, list(10))
some(x, is.character)               # 返回TRUE，翻译为:is some of x is character?
every(x, is.vector)                 # TRUE，翻译为:is every one of x is vector?
x <- sample(10)
detect(x, ~.>8)         # 返回第一个使得谓语为TRUE的内容
head_while(x,~.>5)      # 从头开始返回使得谓语为TRUE的内容
tail_while(x,~.>4)      # 从尾开始返回使得谓语为TRUE的内容
```

# CHPT18 - Model Basics with modelr

## （1）随机参数拟合

本章介绍线性模型的拟合，首先使用最简单粗暴的方法：随机参数拟合，来理解模型的本质。以下为探索过程。

```R
library(tidyverse)
library(modelr)

# 系统自带的模拟数据
=======================
> sim1
# A tibble: 30 x 2
       x     y
   <int> <dbl>
 1     1  4.20
 2     1  7.51
 3     1  2.13
 4     2  8.99
 5     2  10.2 
# ... with 26 more rows
=======================

# 定义随机参数（250 组截距和斜率）
rand_coef <- tibble(
    itc = runif(250, -20, 40),
    slp = runif(250, -5, 5)
)
ggplot(sim1, aes(x, y)) + geom_abline(aes(intercept=itc, slope=slp),
                                     data=rand_coef,
                                     alpha=1/4) + geom_point()

# 衡量模型误差（最小二乘法）
measure_distance <- function(itc, slp, data){
    diff <- data$y - itc - data$x * slp
    sqrt(mean(diff^2))
}

# 计算每组系数的误差
distances <- 1:250
for (i in seq_along(distances)){
    distances[i] <- measure_distance(rand_coef[['itc']][i], 
                                     rand_coef[['slp']][i], 
                                     sim1)
}
rand_coef_dist <- rand_coef %>% mutate(dist=distances)

# 取误差最小的前十组
ggplot(sim1, aes(x,y)) + 
    geom_point(size=2, color="grey30") + 
    geom_abline(aes(intercept=itc, slope=slp, color=-dist),
                data=filter(rand_coef_dist, rank(dist) <= 10)
               )
```

<img src="https://img-blog.csdnimg.cn/20200415054542803.jpg">

## （2）系统拟合方法

```R
# 使用 lm 函数一步完成最小二乘线性拟合
sim1_mod <- lm(y~x, data=sim1)
==========================
> coef(sim1_mod)
(Intercept)           x 
   4.220822    2.051533 
==========================

# 计算模型残差和预测值
sim1 <- sim1 %>% add_residuals(sim1_mod) %>% add_predictions(sim1_mod)
# 可视化拟合结果
ggplot(sim1, aes(x)) + 
    geom_point(aes(y=y)) + 
    geom_line(aes(y=pred), color="green", size=1) + 
    geom_line(aes(y=resid), color="red")
```

<img src="https://img-blog.csdnimg.cn/20200415054712415.jpg">

## （3）其他模型拟合

- `lm(y~x1+x2, data)`
- `lm(y~x1*x2, data)`
- `lm(y~ns(x, n), data)`
- `model_matrix(data, y~x^2+x)`
- `model_matrix(data, y~I(x^2)+x)`
- `model_matrxi(data, y~poly(x, 2))`

## （4）其他模拟拟合函数

- 广义线性模型：`stats::glm()`
- 广义附加模型：`stats::gam()`
- 惩罚线性模型：`glmnet::glmnet()`
- 鲁棒线性模型：`MASS::rlm()`
- 决策树模型：`rpart::rpart()`

# CHPT19 - Model Building

上一章通过模拟数据了解模型拟合函数，本章使用实际数据进行应用。迭代以下分析流程：

- 发现问题
- 问题假设
- 模型选择
- 残差分析
- 新问题

```R
library(modelr)
library(tidyverse)
library(lubridate)
library(nycflights13)
options(na.action = na.warn)
```

## （1）钻石价格与品质的关系

```R
# 首先进行可视化分析，查看每个品质变量与价格的相关关系
ggplot(diamonds, aes(cut, price)) + geom_boxplot()
ggplot(diamonds, aes(color, price)) + geom_boxplot()
ggplot(diamonds, aes(clarity, price)) + geom_boxplot()

# 可以看到在前三个变量中，品质最高的分组都不是均价最高的
# 原因可能是由于低品质的钻石往往重量更大
# 因此需要对重量与价格的关系进行建模分析
ggplot(diamonds, aes(carat, price)) + geom_hex(bins=50, color="grey30")
```

<img src="https://img-blog.csdnimg.cn/20200415054817810.jpg">

```R
# 观察 carat 与 price 的相关趋势，接近指数关系，因此转为对数坐标系
diamonds <- diamonds %>% filter(carat < 2.5) %>%
    mutate(lprice=log2(price), lcarat=log2(carat))
# 在进行可视化，发现对数化之后呈现线性关系

# 进行线性回归
mod <- lm(lprice~lcarat, data=diamonds)
# 添加预测值和残差
diamonds <- diamonds %>% 
    add_predictions(mod, 'lprice_pred') %>% mutate(price_pred=2^lprice_pred) %>% 
    add_residuals(mod, 'lprice_resid')

===============================================================================
> diamonds[c('carat', 'price', 'lprice', 'lcarat', 'lprice_pred', 'price_pred', 'lprice_resid')] %>% head(5)
# A tibble: 5 x 7
  carat price lprice lcarat lprice_pred price_pred lprice_resid
  <dbl> <int>  <dbl>  <dbl>       <dbl>      <dbl>        <dbl>
1 0.23    326   8.35  -2.12        8.63       396.      -0.279 
2 0.21    326   8.35  -2.25        8.41       340.      -0.0587
3 0.23    327   8.35  -2.12        8.63       396.      -0.275 
4 0.290   334   8.38  -1.79        9.19       584.      -0.807 
5 0.31    335   8.39  -1.69        9.35       654.      -0.964 
===============================================================================

# 可视化模型结果
ggplot(diamonds, aes(carat, price)) + geom_hex(bins=50) + 
    geom_line(aes(carat, price_pred), color="green", size=1)
# 残差值的分布
ggplot(diamonds, aes(lcarat, lprice_resid)) + geom_hex(bins=50)
```

<img src="https://img-blog.csdnimg.cn/20200415054916725.jpg">

```R
# 残差中去除了重量与价格的相关性，接下来查看品质变量与价格的关系
ggplot(diamonds, aes(cut, lprice_resid)) + geom_boxplot()
ggplot(diamonds, aes(color, lprice_resid)) + geom_boxplot()
ggplot(diamonds, aes(clarity, lprice_resid)) + geom_boxplot()

# 可以看到在残差中，品质因子与价格变量表现出合理的相关性
```

<img src="https://img-blog.csdnimg.cn/20200415055014496.jpg">

```R
# 直接使用多变量模型进行拟合
mul_mod <- lm(lprice~lcarat+color+cut+clarity, data=diamonds)
diamonds %>% add_predictions(mul_mod)
# 分析略
```

## （2）影响航班每日数量的因素

```R
# 计算每日航班数量的分布
daily <- flights %>% mutate(date=make_date(year, month, day)) %>% 
    group_by(date) %>% summarise(count=n())
ggplot(daily, aes(date, count)) + geom_line()
```

<img src="https://img-blog.csdnimg.cn/20200415055107718.jpg">

```R
# 可视化呈现了与星期有关的周期性
daily <- daily %>% mutate(wday=wday(date, label = T))
ggplot(daily, aes(wday, count)) + geom_boxplot()

# 可以看到周末的航班最少，建模移除以上相关性
mod <- lm(count~wday, data=daily)
grid <- daily %>% data_grid(wday) %>% add_predictions(mod, "count")
ggplot(daily, aes(wday, count)) + geom_boxplot() + 
    geom_point(data = grid, size=4, color="red")
# 添加残差
daily <-daily %>% add_residuals(mod)
daily %>% ggplot(aes(date, resid)) + geom_ref_line(h=0) + geom_line()
```

<img src="https://img-blog.csdnimg.cn/20200415055422669.jpg">

```R
# 可以看到残差里面还遗留了模型没有捕捉到的周期性：
ggplot(daily, aes(date, resid, color=wday)) + geom_ref_line(h=0) + geom_line()
# 按wday分组可以更加清晰地观测到残差里面的模式：
# 例如周六的航班在夏天明显比较多，在秋天比较少
# 某些特定的时间航班会特别多或者特别少，例如某些重要节日
daily %>% filter(resid < -100)
# 图中还存在一个长期的趋势，可以用 geom_smooth 平滑化显示出来
daily %>% ggplot(aes(date, resid)) + geom_ref_line(h=0) + 
    geom_line(color="grey50") + geom_smooth(se=FALSE, span=0.2)
```

<img src="https://img-blog.csdnimg.cn/20200415055532394.jpg">

# CHPT20 - More Models with purrr and broom

**处理大量数据的三种思路：**

- 使用多个简单的模型理解复杂的数据
- 使用列向量在 data.frame 里储存任意的数据结果（nested data.frame）
- 使用 broom 包整理数据

## 示例：分析 gapminder 数据集

```R
library(modelr)
library(tidyverse)
library(gapminder)

# 提出问题：每个国家的 LifeExpectancy 随时间的变化趋势
# 解决思路：参考上一章，通过模型拟合去除线性相关性，最后进行残差分析

# 取新西兰的数据进行分析
new_zealand <- gapminder %>% filter(country == 'New Zealand')
new_zealand %>% ggplot(aes(year, lifeExp)) + geom_line() + ggtitle("Full data =")

# 建模，得到预测值和残差
mod <- lm(lifeExp~year, data=new_zealand)
new_zealand %<>% add_predictions(mod) %>% add_residuals(mod)
new_zealand %>% ggplot(aes(year, pred)) + geom_line() + ggtitle("Linear trend + ")
new_zealand %>% ggplot(aes(year, resid)) + geom_line() + ggtitle("Remaining pattern")
```

<img src="https://img-blog.csdnimg.cn/20200416052115349.jpg">

```R
# 将上述过程打包成函数，利用 purrr::map() 对每个国家重复类似的操作

# 首先对不同国家的数据进行分组
group <- gapminder %>% group_by(country, continent) %>% nest()
===========================================
> group %>% head(5)	# nest()：将组内数据打包
# A tibble: 5 x 3
  country     continent data             
  <fct>       <fct>     <list>           
1 Afghanistan Asia      <tibble [12 x 4]>
2 Albania     Europe    <tibble [12 x 4]>
3 Algeria     Africa    <tibble [12 x 4]>
4 Angola      Africa    <tibble [12 x 4]>
5 Argentina   Americas  <tibble [12 x 4]>
>
> group$data[[1]] %>% head(5)
# A tibble: 5 x 4
   year lifeExp      pop gdpPercap
  <int>   <dbl>    <int>     <dbl>
1  1952    28.8  8425333      779.
2  1957    30.3  9240934      821.
3  1962    32.0 10267083      853.
4  1967    34.0 11537966      836.
5  1972    36.1 13079460      740.
===========================================

# 打包建模函数
country_model <- function(df){
    lm(lifeExp~year, data=df)
}
# 对每个国家进行建模
models <- purrr::map(group$data, country_model)
group <- group %>% mutate(models=models)
# 计算每个国家的预测值和残差
group <- group %>% mutate(resid=map2(data, models, add_residuals))
group <- group %>% mutate(pred=map2(data, models, add_predictions))

# 为了进行残差分析，需要把 nested 数据框转换成标准数据框
resid<-unnest(by_country,resid)
# 可视化
resid %>% ggplot(aes(year, resid)) + 
    geom_line(aes(group=country), alpha=1/3) + geom_smooth(se=FALSE)
# 可以看到部分国家的残差波动较大，进一步可视化每个 continent 的情况
resid %>% ggplot(aes(year, resid)) + 
    geom_line(aes(group=country), alpha=1/3) + facet_wrap(~continent)
# 结果显示非洲和亚洲的拟合效果较差

```

<img src="https://img-blog.csdnimg.cn/20200416052007493.jpg">

## 除了残差，其他判断模型质量的方法

```R
# 使用 glance 函数度量模型效果
==========================================================================
> broom::glance(mod)
# A tibble: 1 x 11
  r.squared adj.r.squared sigma statistic p.value    df logLik   AIC   BIC
      <dbl>         <dbl> <dbl>     <dbl>   <dbl> <int>  <dbl> <dbl> <dbl>
1     0.954         0.949 0.804      205. 5.41e-8     2  -13.3  32.6  34.1
# ... with 2 more variables: deviance <dbl>, df.residual <int>
==========================================================================

# 计算所有模型的统计参数
glances <- group %>% mutate(glances=map(models, broom::glance)) %>% 
    unnest(glances, .drop=T)    # 舍弃除 glances 的其他列表

# 通过对R—squared 进行排序找到拟合不够好的国家模型
glances %>% arrange(r.squared)

# 将所有国家的模型拟合度可视化
# 由于观测值相对小，小于1，而且是一个离散变量，因此使用jitter
glances %>% ggplot(aes(continent, r.squared)) + geom_jitter(width = 0.5)
# 取拟合度差的子集
bad_fit <- filter(glances, r.squared<0.25)
gapminder %>% semi_join(bad_fit, by="country") %>% 
    ggplot(aes(year, lifeExp, color=country)) + geom_line()
# 1992-卢旺达-艾滋病爆发
```

<img src="https://img-blog.csdnimg.cn/20200416051921751.jpg">

## 关于 nested data.frame 的细节补充

```R
# 默认情况下，传入的列表，会被拆成向量分别放到每一列
data.frame(x=list(1:3,2:4))

# 创建 nested data.frame

# 使用 tidyr::nest()
gapminder %>% group_by(country, continent) %>% nest()
gapminder %>% nest(year:gdpPercap)  # nesting year ~ gdpPercp

# 通过向量化函数--mutate()
df <- tribble(~x1, "a,b,c", "d,e,f,g")
df <- df %>% mutate(x2=stringr::str_split(x1, ","))
df %>% unnest()

# 通过 tribble 函数和 map
sim <- tribble(
    ~fun,       ~params,
    "runif",    list(min=-1,max=1),
    "rnorm",    list(sd=5),
    "rpois",    list(lambda=10)
)
sim %>% mutate(sims=invoke_map(fun, params, n=10))
```

# CHPT21 - R Markdown

## 简介

R Markdown provides a unified authoring framework for data science, combining your code, its results, and your prose commentary. R Markdown documents are fully reproducible and support dozens of output formats, like PDFs, Word files, slideshows, and more. R Markdown files are designed to be used in three ways: 

- For communicating to decision makers, who want to focus on the conclusions, not the code behind the analysis. 
-  For collaborating with other data scientists (including future you!), who are interested in both your conclusions, and how you reached them (i.e., the code). 
-  As an environment in which to do data science, as a modern day lab notebook where you can capture not only what you did, but also what you were thinking.

## _.Rmd 文件由以下三部分组成

- An (optional) YAML header surrounded by `---`
- Chunks of R code  surrounded by ` ``` `
- Text mixed with simple text formatting like `#` heading and `_italics_`

## R Markdown 的渲染步骤

When knit the document，Rmarkdown 把文件 send 到 knitr，knitr 生成一个新的 markdown 文件(.md)，然后通过 pandoc 进行 processing，最后生成输出的文件。

## 示例一：Text formatting with Mardown

-------- Text_format.Rmd ---------

```markdown
---
title: "Text Format"
author: "Paradise"
date: "2018年11月22日"
output: html_document
---

Text formatting
------------------------------------------------
斜体格式
*italic* or _italic_

加粗格式
**bold** or __bold__

背景灰色填充格式
`code`

上标与下标
superscript^2^ and superscript~2~


Headings
------------------------------------------------
# 1st Level Header
## 2nd Level Header
### 3rd Level Header

Lists
------------------------------------------------
*   Bulleted list item 1
*   Item 2
    * Item 2a
    * Item 2b
1.  Numbered list item 1
1.  Item 2. The numbers are incremented automatically in the output.

Links and images
-----------------------------------------------
<http://example.com>

添加相关的linked phrase 
[linked phrase](http://example.com)

中括号内为可选标题文字

![optional caption text](../Rplot.jpeg)

Tables
----------------------------------------------

First Header  | Second Header
------------  | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell
```

将上面的 R Markdown 文件渲染成 HTML 文件

```R
rmarkdown::render('./Rmd/Text_format.Rmd', output_flie='Text_format.html')
```
**渲染结果** \| [新标签页查看](/img/in-post/R/Text_format.html)

<div align="center">
<iframe
    style="margin-left:2px; margin-bottom:20px;"
    frameborder="1" scrolling="0"
    width="600" height="360"
    src="/img/in-post/R/Text_format.html">
</iframe>
</div>

## 示例二：Code Chuck

```markdown
---
title: "Code_chunk"
author: "Paradise"
date: "2018年11月22日"
output: html_document
---

Options of Code Chunks
------------------------------
**knitr**提供了近60个参数选项用于自定义chunk的输出

通过以下网址可以查看全部的options:<http://yihui.name/knitr/options/>.

这里展示比较常用的重要的options

​```{r by-name,echo=TRUE,error=TRUE}
x<-runif(10)
y<-runif(10)
x
plot(x,y,type="l")
​```

`说明`
---------------------------------
* 使用eval=FALSE，（当代码块有error仍能）输出代码，但是输出文件不包含执行结果
* 使用include=FALSE，运行代码，但是不在输出文件里显示代码或输出结果
* 使用echo=FALSE，输出执行结果，但是不包含源代码
* 使用message=FALSE或者warning=FALSE，输出文件中不包含warning信息
* 使用result='hide'隐藏打印输出的内容，使用fig.show='hide'隐藏图形输出
* 使用error=TRUE，即使有代码块报错也继续输出文件，在调试阶段使用

Table
-----------------------------
默认情况下数据框输出跟在console的输出格式一样，使用kable函数可以输出表格格式
​```{r name2}
mtcars[1:5,1:10]
knitr::kable(mtcars[1:5,1:10],
             caption = "A knitr table")
​```

`说明`
---------------------------------
* 执行**?knitr::kable**了解很多更细致的自定义表格输出的方法
* 更深入的格式显示方法，可以使用**xtable**,**stargazer**,**pander**,**tables**以及**ascii**等包

Caching
---------------------------------
为了实现可再现性，所有的输出内容都是从空白页面开始构建，以确保代码里包含所有重要信息。但是当代码块里面有计算量较大的指令，那就需要用到缓存：定义cache=TRUE。设置缓存时，计算结果会保存到特定文件，下一次执行时，如果代码块没有改变，则引用该缓存文件。

​```{r raw-data,eval=FALSE}
rawdata<-readr::read_csv("a_very_large_file.csv")
​```

注意例子只是样本代码，设置了eval=FAlSE，不执行

​```{r proccessed-data,cache=TRUE,dependson="raw_data",eval=FALSE}
processing the terribly large data
​```

此处注意，如果没有定义dependson="raw_data"，那么即使读取的csv文件改变了，
只要‘processing’没有改变，仍然不会重新执行该代码块，而是直接使用缓存。

如果担心"a_very_large_file.csv"文件内容本身发生改变，导致后续的代码使用缓存而出错，可以在raw_data代码块增加cache.extra=file.info("file_name")这样可以检查文件的相关信息，包括最后一次修改的时间。

​```{r clean_up,eval=FALSE}
knitr::clean_cache()
​```
使用上述指令可以清除所有的缓存（当缓存的文件越来越复杂混乱的时候）

Global Options
----------------------------
`关于全局设置的内容在第五部分的script中出现`

Inline Code
-----------------------------
直接在文本中间引用并嵌入R代码，knit之后以执行结果的形式显示

For example,we have data about **`r nrow(diamonds)`** diamonds.

在文本中嵌入数字的时候，最好设置精确度，以及在大数中插入逗号(使用**format**函数)
​```{r,collapse=FALSE}
a<-1234567
format(a,big.mark = ",")
b<-0.1234567
format(b,digits = 2)
​```
```

**渲染结果** \| [新标签页查看](/img/in-post/R/Code_chunk.html)

<div align="center">
<iframe
    style="margin-left:2px; margin-bottom:20px;"
    frameborder="1" scrolling="0"
    width="600" height="360"
    src="/img/in-post/R/Code_chunk.html">
</iframe>
</div>

# CHPT22 - Graphics for Communication with ggplot2

此前介绍的绘图操作仅限于 EDA 过程中的可视化，本章介绍用于 Report 的可视化技巧，即图片的修饰和美化。

## 标题与副标题、图片脚注、坐标轴与图例标题，数学公式

```R
library(tidyverse)
base <- ggplot(mpg, aes(displ, hwy)) + 
    geom_point(aes(color=class)) + geom_smooth(se=FALSE)

# 使用 labs 函数在图中添加标题和图片标签
base + labs(
    title="Fuel efficiency generally decreases with engine size",
    subtitle="Two seaters(sport cars) are an exception because of light weight",
    caption="Data from fueleconomy.gov"
)

# 使用 labs 函数自定义 axis 和 legend 标题
base + labs(
    x="Engine displacement (L)",
    y="Highway fuel economy (mpg)",
    color="Car type"
)

# 使用 quote 函数输出数学公式格式
df = data.frame(x=rnorm(30), y=rnorm(30))
df %>% ggplot(aes(x, y)) + geom_point() + labs(
    x = quote(sum(x[i]^2, i==1, n)),
    y = quote(alpha + beta + frac(delta, beta))
)
```

<img src="https://img-blog.csdnimg.cn/20200416051747732.jpg">

## 点注释

```R
# 使用 geom_text 函数添加点注释
bestcar <- mpg %>% group_by(class) %>% filter(row_number(desc(hwy)) == 1)
base + geom_text(aes(label=model), data=bestcar)

# 使用 geom_label 函数添加点注释（注释带有边框）
base + geom_label(
    aes(label=model), data=bestcar, alpha=0.5,
    nudge_y=2	# 用于移动注释和点的相对位置
)

# 上述两者皆有注释重叠情况，可使用 geom_label_repel 函数优化
base + geom_point(size=3, alpha=1, data=bestcar) + 
    ggrepel::geom_label_repel(aes(label=model), data=bestcar)

# 使用 label 代替图中的 legend
class_avg <- mpg %>% group_by(class) %>% 
    summarise(displ=median(displ), hwy=median(hwy))
ggplot(mpg, aes(displ, hwy, color=class)) + 
    ggrepel::geom_label_repel(
        aes(label=class), data=class_avg,
        size=5, label.size=0, segment.color = NA
    ) + 
    geom_point() + 
    theme(legend.position="none")
```

<img src="https://img-blog.csdnimg.cn/20200416051636132.jpg">

## 单个注释

```R
# 使用 geom_text 函数添加单个注释
anno1 <- mpg %>% summarise(displ=max(displ), hwy=max(hwy), label='Annotaions')
base + geom_text(
    aes(label=label), data=anno1, vjust='top', hjust='right'
)
# 使用 str_wrap 函数对注释文本进行排版
anno2 <- tibble(displ=Inf, hwy=Inf, 
                label='abcdef一二三四五六' %>% stringr::str_wrap(width=10)
                )
base + geom_text(
    aes(label=label), data=anno2, vjust='top', hjust='right'
)

# 其他常用添加注释（非文本）的函数
geom_hline; geom_vline      # 辅助线
geom_rect                   # 矩形框
geom_segement               # 利用 arrow 参数可以添加 箭头
```

<img src="https://img-blog.csdnimg.cn/20200416051546278.jpg">

## 坐标刻度、图例样式

```R
# Scales
base <- ggplot(mpg, aes(displ, hwy)) + geom_point(aes(color=class))
# R 会自动调整 scales，当运行 base，相当于：
base + scale_x_continuous() + scale_y_continuous() + scale_color_discrete()

# 调整y轴示数的 breaks，隐藏x轴示数
base + 
    scale_y_continuous(breaks=seq(15, 40, by=5)) + 
    scale_x_continuous(labels = NULL)

# 使用 theme 和 guides 调整 legend 的样式
base + 
    theme(legend.position="bottom") + 
    guides(
        color=guide_legend(nrow=1, override.aes=list(size=4))
    )
```

<img src="https://img-blog.csdnimg.cn/20200416051416262.jpg">

## 坐标转换、坐标缩放

```R
# 坐标的对数化
# 变量直接对数化，坐标刻度也跟随改变
ggplot(diamonds, aes(log(carat), log(price))) + geom_hex()
# 转换坐标轴，坐标刻度不变
ggplot(diamonds, aes(carat, price)) + geom_hex() + 
    scale_x_log10() + scale_y_log10()

# 设置坐标轴取值范围
base + coord_cartesian(xlim = c(5,7), ylim = c(10,30))

# 不同图片使用相同的scale方便进行比较
suv <- mpg %>% filter(class=="suv")
compact <- mpg %>% filter(class=="compact")
# 借助 limits 参数以及 range、unique 函数创建统一坐标尺度
x_scale <- scale_x_continuous(limits=range(mpg$displ))
y_scale <- scale_y_continuous(limits=range(mpg$hwy))
color_scale <- scale_color_discrete(limits=unique(mpg$drv))
# 使用统一的坐标尺度
ggplot(suv, aes(displ, hwy, color=drv)) + geom_point() + 
    x_scale + y_scale + color_scale
ggplot(compact, aes(displ, hwy, color=drv)) + geom_point() + 
    x_scale + y_scale + color_scale
```

<img src="https://img-blog.csdnimg.cn/20200416051306787.jpg">

## 颜色风格

```R
# 颜色风格的设置（scale_color_brewer）
base <- ggplot(mpg, aes(displ, hwy, color=class)) + geom_point()
all_scales <- list(
    style1=c("YlOrRd","YlOrBr","YlGnBu","YlGn","Reds","RdPu",
             "Purples","PuRd","PuBuGn","PuBu","OrRd","Oranges",
             "Greys","Greens","GnBu","BuPu","BuGn","Blues"),
    style2=c("Set1","Set2","Set3",
             "Pastel2","Pastel1","Paired","Dark2","Accent"),
    style3=c("Spectral","RdYlGn","RdYlBu",
             "RdGy","RdBu","PuOr","PrGn","PiYG","BrBG")
)
base + scale_color_brewer(palette=all_scales[[3]][6])

# 手动设置颜色风格（scale_color_manual）
presidential %>% mutate(id=33+row_number()) %>% 
    ggplot(aes(start, id, color=party)) + geom_point() + 
    geom_segment(aes(xend=end, yend=id)) + 
    scale_color_manual(
        values = c(Republican="red", Democratic="blue")
    )

# 使用 viridis 包优化颜色风格（scale_fill_viridis）
df <- tibble(x=rnorm(1000), y=rnorm(1000))
# ggplot(df, aes(x,y)) + geom_hex() + coord_fixed()
ggplot(df,aes(x,y)) + geom_hex() + viridis::scale_fill_viridis() + coord_fixed()

# 使用 ggplot2 的内置主题
base+theme_dark(base_line_size=0)
```

<img src="https://img-blog.csdnimg.cn/20200416051128530.jpg">

# CHPT23 - R Markdown Formats

```R
# (1)Introduction
# 在render函数中设置输出文件格式
rmarkdown::render("example.Rmd",output_format = "word_document")
# 或者在knit按钮的下拉菜单中选择knit的格式

# (2)Output Options
# 查看输出html文件时可以设置那些参数(Then cmd:Ctrl+3)
?rmarkdown::html_document()
# 使用expanded output field改写default的参数(Rmd 文档)
output: 
    html_document: 
        toc: true
        toc_float: true
# 输出多个不同格式的文件
output: 
    html_document: 
        toc: true
        toc_float: true
    pdf_document: default

# (3)Documents
# 支持的文件格式：
# pdf、word、odt(OpenDocument Text)、rtf(Rich Text Format)、md、github
# 使code chunk隐藏的方法：
knitr::opts_chunk$set(echo = FALSE)
# 在html文件中，可以通过option设置(点击可以使代码出现)
output: 
    html_document: 
        code_folding: hide

# (4)Notebooks
# html_document主要用于与决策者交流，而html_notebook用于与其他analysist交流

# (5)Presentations(类似PPT的功能)
# RMarkdown支持的三种展示格式：
ioslides_presentation()
slidy_presentation()
beamer_presentation()
# 另外两种由package提供的格式(revealjs与rmdshower包)

# (6)Dashboards(仪表盘)
# 用于可视化地和快速地交流大量的信息
# 通过headers控制输出的布局：
#Each level 1 header(#) begins a new page in the dashboard
#Each level 2 header(##) begins a new column
#Each level 3 header(###) begins a new row

# 具体案例：
render("R4DS5_dashboard.Rmd",output_file = "dashboard.html")
# learn more about flexdashboard:
# see <http://rmarkdown.rstudio.com/flexdashboard/>
```

**渲染结果** \| [新标签页查看](/img/in-post/R/dashboard.html)

<div align="center">
<iframe
    style="margin-left:2px; margin-bottom:20px;"
    frameborder="1" scrolling="0"
    width="600" height="360"
    src="/img/in-post/R/dashboard.html">
</iframe>
</div>

```R
# (7)Interactivity(交互)
# 任何html format(document/notebook/presentation/dashboard)都可以包含交互性内容

# htmlwidgets(使用leaflet包产生交互性网页)
library(leaflet)
leaflet()%>%
    setView(110.922,21.603,zoom = 2)%>%
    addTiles()%>%
    addMarkers(110.922,21.603,popup = "HOME")
# 其他提供htmlwidgets的包：dygraphs/DT/rthreejs/DiagrammeR
# learn more about htmlwidgets: <http://www.htmlwidgets.org/>

# shiny
# htmlwidgets提供客户端的互动，与R完全脱离，内部使用HTML和JavaScript进行控制
# 而Shiny允许使用R代码生成互动页面
# 在Rmd的header中call shiny
title: "shiny"
output: html_document
runtime: shiny
# 用Input函数添加互动性内容
library(shiny)
textInput("name", "What is your name?")
numericInput("age", "How old are you?",NA,min=0,max=150)
# 具体效果见html文件：
render("R4DS5_shiny.Rmd",output_file = "shiny.html")
# learn more about shiny<http://shiny.rstudio.com/>

# (8)Websites
# learning more: <http://bit.ly/RMarkdownWebsites>

# (9)Other Formats
# The bookdown package makes it easy to write books.
# Read <Authoring Book with R Markdown>
# see <http://www.bookdown.org>

# The prettydoc package provides lightweight document formats with attractive themes

# The rticles package 
# see more: <http://rmarkdown.rstudio.com/formats.html>
# create your own formats: <http://bit.ly/CreateNewFormats>
```

**渲染结果** \| [新标签页查看](/img/in-post/R/shiny.html)

<div align="center">
<iframe
    style="margin-left:2px; margin-bottom:20px;"
    frameborder="1" scrolling="0"
    width="600" height="360"
    src="/img/in-post/R/shiny.html">
</iframe>
</div>

# CHPT24 - R Markdown Workflow

**SKIP**

------------------------------

**END**
