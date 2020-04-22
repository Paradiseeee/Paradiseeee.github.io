# Python DataScience CookBook -- Learning Notes

-- 2019-04-21
-- subtitle
-- 机器学习、python、数据分析

> 教材介绍：<https://book.douban.com/subject/26630181/>


## 简介

这本书比较冷门，因为需要快速上手 python 机器学习，随便找本书来学习一下。虽然内容比较浅显，不够深入，但是作为快速上手的教材很好用。其实整本书就是相当于 scikit-learn 的一个帮助文档而已，没有扯什么原理性的东西。如果有一定基础，要熟悉 scikit-learn，其实就是拿速查表刷一遍就行了：

<img src="Python-Scikit_Learn.png">

本书前三章介绍 Python 编程的基础和数据处理、数据分析的基础，比较熟悉的内容，直接跳过。记录一下后面几章的学习笔记。文中涉及的算法原理可以参考另一篇笔记：[R 统计学习（ISLR）-- Learning Notes](https://paradiseeee.github.io/2019/01/10/R-%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0-ISLR-Learning-Notes-I/)。

- 第四章：数据分析 -- 深入理解
    - 抽取主成分
    - 使用核PCA
    - 使用奇异值分解抽取特征
    - 用随机映射进行数据降维
    - 用NMF分解特征矩阵
- 第五章：数据挖掘
    - 使用距离度量
    - 使用核方法
    - k-means聚类
    - 向量量化
    - 在单变量数据中找出异常点
    - 使用局部异常因子方法发现异常点
- 第六章：机器学习（1）
    - 为建模准备数据
    - 最邻近算法
    - 朴素贝叶斯分类
    - 构建决策树解决多分类问题
- 第七章：机器学习（2）
    - 回归方法预测实数值
    - 岭回归
    - lasso
    - L1 缩减 和 L2 缩减交叉验证
- 第八章：集成方法
    - 装袋法
    - 提升法
    - 梯度提升
- 第九章：生长树
    - 随机森林
    - 超随机树
    - 旋转森林
- 第十章：大规模机器学习 -- 在线学习
    - 用感知器作为在线学习算法
    - 用梯度下降解决回归问题
    - 用梯度下降解决分类问题



## 第四章 数据分析 -- 深入理解

### **（1）主成分分析**

- 对于多变量问题，进行 PCA 降维只有很小的信息损失。
- 对于一维数据，使用方差衡量数据的变异情况；对于多维数据，使用协方差矩阵。
- 示例：在 iris 数据集上进行 PCA 降维：
    - 数据标准化：均值为 0，方差为 1
    - 计算数据的相关矩阵和单位标准差偏差值
    - 将相关矩阵分解成特征向量和特征值
    - 根据特征值的大小，选择 Top-N 个特征向量
    - 投射特征向量矩阵到一个新的子空间
- 选取特征值的标准：
    - 特征值标准：特征值为 1，意味至少可以解释一个变量，至少为 1 才能选取
    - 变异解释比 PVE：一般以累计值为标准，从 Top-N 主成分累计到接近 100% 


```python
import scipy
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale

# iris 数据集：3个分类，4维特征
iris = load_iris()
X, Y = iris['data'], iris['target']
# 标准化：由于 PCA 为无监督方法，只需标准化 features
x_s = scale(X, with_mean=True, with_std=True ,axis=0)

# 计算相关矩阵：
x_corr = np.corrcoef(x_s.T)
# 从相关矩阵中计算特征值和特征向量：
eigenvalue, right_eigenvector = scipy.linalg.eig(x_corr)
# 选择 Top-2 特征向量（eig 函数输出降序排列）
w = right_eigenvector[:, 0:2]
# 使用特征向量作为权重进行PCA降维（投影到特征向量方向）
x_rd = x_s.dot(w)

# 画出新的特征空间的散点图
plt.figure(facecolor='#ffffff')
plt.scatter(x_rd[:,0], x_rd[:,1], c=Y)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# 按照变准选取特征值
df = pd.DataFrame(
    np.random.randn(4,3), 
    columns=['Eigen Values', 'PVEs', 'Cummulative PVE'],
    index=pd.Index([1,2,3,4], name='Principal Component')
    )

cum_pct, var_pct = 0, 0
for i, eigval in enumerate(eigenvalue):
    var_pct = round((eigval / len(eigenvalue)), 3)
    cum_pct += var_pct
    df['Eigen Values'][i+1] = eigval
    df['PVEs'][i+1] = var_pct
    df['Cummulative PVE'][i+1] = cum_pct

df.plot()
plt.show()
# 可以看到前两个主成分解释了 95.9% 的变异
```

<img src="iris-pca-result.jpg">

### **（2）使用核 PCA**

核 PCA 是 PCA 的非线性扩展，当分类不是线性可分的，在进行 PCA 时通过核函数转换数据点，将数据映射到核空间，最后在核空间进行线性 PCA。

```python
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

# 使用 make_circles 生成一个非线性数据集
np.random.seed(0)
# 二维特征，先验二分类，非线性：
X, Y = make_circles(n_samples=400, factor=0.2, noise=0.02)

# 可视化结果
def visualization(X, Y, title):
    '''可视化前两个主成分（一共有几百个）'''
    plt.figure(figsize=(6,6))
    plt.title(title)
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.xlabel('$Component_1$'); plt.ylabel('$Component_2$')
    plt.show()

# PCA 函数集成了上一节中的拟合预测等运算

# 首先使用线性 PCA
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
visualization(x_pca, Y, 'Linear PCA')

# 使用核 PCA（径向基核函数）
kpca = KernelPCA(kernel='rbf', gamma=10)
kpca.fit(X)
x_kpca = kpca.transform(X)
visualization(x_kpca, Y, 'Kernel PCA')
```

<img src="circle-data-kernel-pca-result.jpg">

### **（3）使用奇异值分解提取特征**

**奇异值分解（Singular Value Decomposition, SVD）：**将一系列相关变量转换成不相关的变量，实现降维。SVD 常用于文本挖掘，用来挖掘语义关联。

和 PCA 不同，SVD 直接作用于原始数据矩阵，用较低维度的数据得到原始数据的最佳近似。本质上 SVD 不是一种机器学习方法，而是一种矩阵分解技术。公式表示为：`A = U * S * V.T`，其中 U、V 分别成为“左、右奇异向量”，S 为奇异值。

```python
import scipy
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale

# Load dataset
iris = load_iris()
X, Y = iris['data'], iris['target']
# Standardize（如果所有变量度量单位一致，可以不必进行缩放，只需中心化）
x_s = scale(X, with_mean=True, with_std=False, axis=0)

# 通过 SVD 提取特征
U, S, V = svd(x_s, full_matrices=False)
# 选用前两个奇异向量表示原始数据矩阵
x_t = U[:, :2]

# 可视化降维后的数据集
plt.figure(figsize=(5,5))
plt.scatter(x_t[:,0], x_t[:,1], c=Y)
plt.xlabel('$Feature_1$'); plt.ylabel('$Feature_2$')
plt.show()
```

<img src="iris-svd-result.png">

### **（4）用随机映射进行数据降维**

PCA 和 SVD 的运算代价高昂，随机映射方法运算速度更快。根据 *Johnson-Linden Strauss* 定理的推论，从高维到低维的 Euclidean Space 的映射是存在的，可以使点到点的距离保持在一个 epsilon 的方差内。随机映射的目的就是保持任意两点之间的距离，同时降低数据的维度。

```python
from sklearn.metrics import euclidean_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection

# 处理 20 个新闻组的文本数据，采用高斯随机映射
# 高斯随机矩阵是从正态分布 N(0, 1000^-1) 中采样生成的，1000是结果的维度

# 使用 sci.crypt 分类，将文本数据转换为向量表示
data = fetch_20newsgroups(categories='sci.crypt')
# 下载完会本地化，储存进 sklearn 模块

# 从 data 中创建一个 词-文档 矩阵，词频作为值
vectorizer = TfidfVectorizer(use_idf=False)
vector = vectorizer.fit_transform(data.data)
print(f'The Dimension of Original Data: {vector.shape}')
# 使用随机映射降维到 1000 维
gauss_proj = GaussianRandomProjection(n_components=1000)
gauss_proj.fit(vector)
# 将原始数据转换到新的空间
vector_t = gauss_proj.transform(vector)
print(f'The Dimension of Transformed Data: {vector_t.shape}')
# 检验是否保持了数据点的距离
org_dist = euclidean_distances(vector)
red_dist = euclidean_distances(vector_t)
diff_dist = abs(org_dist - red_dist)

# 上面的 diff_dist 返回一个 n x n 方阵，绘制成热力图：
plt.figure(figsize=(8, 8))
plt.pcolor(diff_dist[0:100, 0:100])
plt.colorbar()
plt.show()
```

<img src="newsgroup-random-projection-result.jpg">

### **（5）使用 NMF 分解特征矩阵**

前文使用主成分分析和矩阵分解技术进行降维，Non-negative Matrix Factorization（NMF）采用协同过滤算法进行降维。原理：
输入 `m*n` 矩阵 A，分解为 `A_dash(m*d)` 和 `H(d*n)`，即：`A(m*n) = A_dash * H`。约束条件：最小化 `| A - A_dash * H|^2` 。

```python
# 数据集：电影影评数据
ratings = [
    [5., 5., 4.5, 4.5, 5., 3., 2., 2., 0., 0.],
    [4.2, 4.7, 5., 3.7, 3.5, 0., 2.7, 2., 1.9, 0.],
    [2.5, 0., 3.3, 3.4, 2.2, 4.6, 4., 4.7, 4.2, 3.6],
    [3.8, 4.1, 4.6, 4.5, 4.7, 2.2, 3.5, 3., 2.2, 0.],
    [2.1, 2.6, 0., 2.1, 0., 3.8, 4.8, 4.1, 4.3, 4.7],
    [4.7, 4.5, 0., 4.4, 4.1, 3.5, 3.1, 3.4, 3.1, 2.5],
    [2.8, 2.4, 2.1, 3.3, 3.4, 3.8, 4.4, 4.9, 4.0, 4.3],
    [4.5, 4.7, 4.7, 4.5, 4.9, 0., 2.9, 2.9, 2.5, 2.1],
    [0., 3.3, 2.9, 3.6, 3.1, 4., 4.2, 0.0, 4.5, 4.6],
    [4.1, 3.6, 3.7, 4.6, 4., 2.6, 1.9, 3., 3.6, 0.]
]
movie_dict = {
    1: 'Star Wars',
    2: 'Matrix',
    3: 'Inception',
    4: 'Harry Potter',
    5: 'The hobbit',
    6: 'Guns of Navarone',
    7: 'Saving Private Ryan',
    8: 'Enemy at the gates',
    9: 'Where eagles dare',
    10: 'Great Escape'
}
```

```python
from collections import defaultdict
from sklearn.decomposition import NMF

# 以下是模拟推荐系统的问题，通过用户对电影的评分，预测未知电影的评分。
A = np.asmatrix(ratings, dtype=float)
nmf = NMF(n_components=2, random_state=1)
A_dash = nmf.fit_transform(A)

# 检查降维后的矩阵
for i in range(A_dash.shape[0]):
    print("User id = %d, comp_1 score = %0.2f, comp_2 score = %0.2f" % (i+1, A_dash[i][0], A_dash[i][1]))
plt.figure(figsize=(5,5))
plt.title("User Concept Mapping")
plt.scatter(A_dash[:,0], A_dash[:,1])
plt.xlabel("Component 1 Score"); plt.ylabel("Component 2 Score")
plt.show()

# 检查成分矩阵
F = nmf.components_
plt.figure(figsize=(5,5))
plt.title("Movie Concept Mapping")
plt.scatter(F[0,:], F[1,:])
plt.xlabel("Component 1 Score"); plt.ylabel("Component 2 Score")
for i in range(F[0,:].shape[0]):
    plt.annotate(movie_dict[i+1], (F[0,:][i], F[1,:][i]))
plt.show()
```

<img src="movies-nmf-result.jpg">


## 第五章 数据挖掘

### **（1）使用距离度量**

- 距离度量函数需要满足的条件：
    - 输出是非负的
    - 当且仅当 X=Y 时输出为零
    - 距离是对称的 d(X, Y) = d(Y, X)
    - 遵循三角不等式：d(X, Y) >= d(X, Z) + d(Z, Y)
- 常用度量方法：
    - 欧氏距离：
        - 欧几里得空间：空间中的点是由实数值组成的向量
        - 欧几里得空间的点之间的物理距离成为欧氏距离，也即 L2 范数：
            - `d([x1, x2, ..., xn], [y1, y2, ..., yn]) = sqrt{sum[(xi-yi)^2]}`
    - 余弦距离：
        - 欧几里得空间，以及由整数或布尔值组成的向量空间，都可应用余弦距离
        - 用两个向量夹角的余弦值度量
        - 表达式：`np.dot(X, Y) / np.sqrt(np.dot(X, X) * np.dot(Y, Y))`
    - Jaccard 距离：
        - 给定输入向量的集合，他们交集和并集的大小之比称为Jaccard 系数，1 减去 Jaccard 系数即为 Jaccard 距离
    - Hamming 距离：
        - 对于两个位类型的数据，汉明距离就是这两个两个向量之间不同的位的数量
    - Manhattan 距离：
        - 又称 City Block Distance，也就是用 L1 Norm 度量的距离

```python
euclidean_distance = lambda x, y: np.sqrt(np.sum(np.power((x-y), 2)))
LrNorm_distance = lambda x, y, r: np.power(np.sum(np.power((x-y), 2)), 1/r)
cosine_distance = lambda x, y: np.dot(x,y) / np.sqrt(np.dot(x,x) * np.dot(y,y))
jaccard_distance = lambda x, y: 1 - len(set(x).intersection(set(y))) / len(set(x).union(set(y)))
hamming_distance = lambda x, y: sum([i != j for i, j in zip(x, y)])
```

### **（2）使用核函数**

当数据非线性时，要使用线性模型，需要进行复杂的运算进行线性化。使用核函数可以更便捷地处理非线性数据。核函数的数学定义：`k(x_i, j_i) = <phi(x_i), phi(x_j)>`。这里的 `x_i` 和 `x_j` 为输入向量，`< >` 表示向量点积，映射函数 `phi( )` 将输入向量映射到一个新的空间。

例如设定映射函数为 `phi(x1, x2, x3) = (x1^2, x2^2, x3^2, x1x2, x1x3, x2x1, x2x3, x3x1, x3,x2)`，就可以将输入变量映射到新的空间。

```python
# 示例：在三维数据上简单应用核函数
x = np.array([10, 20, 30])
y = np.array([8, 9, 10])

# 定义核函数：
def mapping_function(x):
    output_list = []
    for i in range(len(x)):
        for j in range(len(x)):
            output_list.append(x[i]*x[j])
    return np.array(output_list)

# 应用核函数
x_t = mapping_function(x)
y_t = mapping_function(y)
print(x_t)
print(y_t)
print(np.dot(x_t, y_t))
```

### **（3）K Means  聚类**

K Means 聚类原理参考“R 统计学习笔记”。

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 随机数据
x_1 = np.random.normal(loc=0.2, scale=0.2, size=(100, 100))
x_2 = np.random.normal(loc=0.9, scale=0.1, size=(100, 100))
x = np.r_[x_1, x_2]

# 将不同 K 值的聚类打包成函数
def form_clusters(x, k):
    km = KMeans(n_cluster=k, init='random')
    km.fit(x)
    return silhouette_score(x, km.labels_)

# 进行参数选择
silh_scores = []
for i in range(10):
    ss = form_clusters(x, i+2)
    silh_scores.append(ss)
# 当 K=2 时轮廓系数最大，聚类最好
```

### **（4）学习向量量化**

Learning Vector Quantization（LVQ）是一种无模型数据点聚类方法。与其他方法不同，它无法解释响应变量和预测变量直接的关系。在实际应用中，作为黑箱方法应用。

LVQ 是一种在线学习方法，每次只处理一个数据点。步骤如下：

- 为数据集里的每个类型选择 K 个原型向量。例如对于二分类问题，每个分类选择两个原型向量，则需要设置 4 个原型向量。它们从数据集中随机地选取；
- 接着确定一个 epsilon 值，进行循环，直到 epsilon 变为 0 或预设的阈值；
- 在上述的每次循环中，都采样一个输入点，采用欧氏距离找出其最近的原型向量。

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
#from sklearn.metrics.pairwise import euclidean_distances

euclidean_distance = lambda x, y: np.sqrt(np.sum(np.power((x-y), 2)))

# 定义一个类来保存原型向量
class Prototype(object):

    def __init__(self, class_id, p_vector, epsilon):
        self.class_id = class_id
        self.p_vector = p_vector
        self.epsilon = epsilon

    def update(self, u_vector, increment=True):
        step = self.epsilon * (u_vector - self.p_vector)
        if increment:
            # 将原型向量靠近输入向量
            self.p_vector = self.p_vector + step
        else:
            # 将原型向量远离输入向量
            self.p_vector = self.p_vector - step

# 寻找离给定向量最近的原型向量
def find_closest(in_vector, proto_vectors):
    closest = None
    closest_distance = 1e5
    for p_v in proto_vectors:
        distance = euclidean_distance(in_vector, p_v.p_vector)
        if distance < closest_distance:
            closest_distance = distance
            closest = p_v
    return closest

# 找出最近的原型向量的类别ID
find_class_id = lambda test_vector, p_vectors: find_closest(test_vector, p_vectors).class_id

# if __name__ == "__main__":
# Loading dataset
iris = load_iris()
X = iris['data']; Y = iris['target']

# 最小最大缩放
minmax = MinMaxScaler()
x = minmax.fit_transform(X)

# 为每个类选择R个原型向量
p_vectors = []
for i in range(3):
    x_subset = x[np.where(Y==i)]
    # 获取R个随机下标，介于0-50
    samples = np.random.randint(0, len(x_subset), 2)
    # 选择原型向量
    for sample in samples:
        s = x_subset[sample]
        p = Prototype(i, s, epsilon=0.9)
        p_vectors.append(p)
print("class id | Initial prototype vector")
for p_v in p_vectors:
    print(p_v.class_id, '\t\t |', p_v.p_vector)

# 利用已有的数据点，执行循环调整原型向量，对新的点进行分类
epsilon = 0.9; delta_epsilon = 0.001
while epsilon >= 0.01:
    # 随机采样一个训练实例
    rnd_i = np.random.randint(0, 149)
    rnd_s = x[rnd_i]
    target_y = Y[rnd_i]
    # 为下一次循环减少epsilon
    epsilon = epsilon - delta_epsilon
    # 查找与给定点最近的原型向量
    closest_pvector = find_closest(rnd_s, p_vectors)
    # 更新最近的原型向量
    if target_y == closest_pvector.class_id:
        closest_pvector.update(rnd_s)
    else:
        closest_pvector.update(rnd_s, False)
    closest_pvector.epsilon = epsilon
print("class_id | Final Prototype Vector")
for p_v in p_vectors:
    print(p_v.class_id, '\t\t |', p_v.p_vector)

# 把原型向量用于预测
y_pred = [find_class_id(instance, p_vectors) for instance in X]
# 根据上面获得的原型向量预测类别
print(classification_report(Y, y_pred, 
        target_names=['Setosa', 'Verssicolour','Virginaca']))
```

### **（5）在单变量数据中找出异常点**

在单变量中寻找异常数据的几种常用方法：
- 绝对中位差
- 平均值加或减去 3 倍标准差

```python
# 创建100个数据点，其中10%是outliers/
n_samples = 100
fraction_of_outliers = 0.1
number_outliers = int(fraction_of_outliers * n_samples)
number_inliers = n_samples - number_outliers

# 创建数据
normal_data = np.random.randn(number_inliers, 1)
print(" | Mean: %0.2f | Standard Deviation: %0.2f |"\
        % (np.mean(normal_data), np.std(normal_data)))
outlier_data = np.random.uniform(low=-9, high=9, size=(number_outliers, 1))
data = np.r_[normal_data, outlier_data]
print(f"Shape of total data: {data.shape}")

# 显示数据
plt.figure(figsize=(5, 5))
plt.title("Input Data Points")
plt.scatter(range(len(data)), data, c='b')
plt.show()
```

```python
'''
mad 对应绝对中卫差方法，std 对应 3 倍标准差方法
'''
def show_result(data, method):

    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)

    # 定义上限和下限
    if method == 'mad':
        b = 1.4826
        mad = b * np.median(np.abs(data - median))
        lower_limit = median - 3 * mad
        upper_limit = median + 3 * mad
    elif method == 'std':
        b = 3
        lower_limit = mean - b * std
        upper_limit = mean + b * std

    # 寻找outliers
    outliers = []
    outliers_index = []
    for i in range(len(data)):
        if data[i] > upper_limit or data[i] < lower_limit:
            outliers.append(data[i])
            outliers_index.append(i)

    # 绘图显示异常点
    plt.figure(figsize=(5,5))
    plt.title(f"Outliers using {method}")
    plt.scatter(range(len(data)), data, c='b')
    plt.scatter(outliers_index, outliers, c='r')
    plt.show()

show_result(data, method='mad')
show_result(data, method='std')

# 可见使用中位值作为评估值比使用均值更加健壮，不容易干扰。
```

<img src="find-outliers-result.jpg">

### **（6）使用局部异常因子方法发现异常点**

Local Outlier Factor（LOF）对数据的局部密度和邻居进行比较，判断这个数据点是否属于相似的密度区域。适用簇的个数未知，簇的密度和大小各不相同的数据中筛选异常点。这种算法思想源自 KNN。

相关术语：
- 对象 P 的 K 距离：对象 P 与其第 K 个最邻近的点的距离，K为自由参数
- P 的 K 距离邻居：距离小于或等于 P 的 K 距离的对象的集合 Q
- P 到 Q 的可达距离（reachability distance）：P 与 其第 K 个最近邻的距离，和 P 与 Q 之间的距离，两者之间的最大值
- P 的局部可达密度（Local Reachability Density, LRD）：K 距离邻居和 K 与其邻居的可达距离之和的比值
- P 的局部异常因子（LOF）：P 与它的 K 最近邻的局部可达距离的比值的平均值

```python
import headq
from collections import defaultdict
from sklearn.metrics import pairwise_distances

# generate data
instances = np.matrix([[0,0], [0,1], [1,1], [1,0], [5,0]])
x = np.squeeze(np.array(instances[:,0]))
y = np.squeeze(np.array(instances[:,1]))

'''计算两两之间的距离'''
k = 2
dist = pairwise_distances(instances, metric='manhattan')

'''计算K距离'''
k_distance = defaultdict(tuple)
for i in range(instances.shape[0]):
    # 获得当前点与其他各点之间的距离
    distances = dist[i].tolist()
    # 获得K最近邻及其索引
    kneighbors = heapq.nsmallest(k+1, distances)[1:][k-1]
    neighbors_idx = distances.index(kneighbors)
    # 每个点的第K个最近邻以及距离
    k_distance[i] = (kneighbors, neighbors_idx)

'''计算K距离邻居'''
def all_indices(value, inlist):
    out_indices = []
    idx = -1
    while True:
        try:
            idx = inlist.index(value, idx+1)
            out_indices.append(idx)
        except ValueError:
            break
    return out_indices

k_distance_neig = defaultdict(list)
for i in range(instances.shape[0]):
    distances = dist[i].tolist()
    print("k distance neighbourhood", i)
    print(distances)
    # 获取 1-K 最邻近
    kneighbors = heapq.nsmallest(k+1, distances)[1:]
    print(kneighbors)
    print(set(kneighbors))

    kneighbors_idx = []
    # 获取 K 里最小的元素的索引
    for x in set(kneighbors):
        kneighbors_idx.append(all_indices(x, distances))
    # 将 列表列表 转化为 列表
    kneighbors_idx = [item for sublist in kneighbors_idx for item in sublist]
    # 对每个点保存其k距离邻居
    k_distance_neig[i].extend(zip(kneighbors, kneighbors_idx))

'''计算可达距离和局部可达密度LRD'''
lrd = defaultdict(float)
for i in range(instances.shape[0]):
    # LRD的分子，也就是K距离邻居的个数
    no_neighbors = len(k_distance_neig[i])
    # 可达距离求和作为分母
    denom_sum = 0
    for neigh in k_distance_neig[i]:
        denom_sum += max(k_distance[neigh[1]][0], neigh[0])
    lrd[i] = no_neighbors / (1. * denom_sum)

'''计算LOF'''
lof_list = []
for i in range(instances.shape[0]):
    lrd_sum, rdist_sum = 0, 0
    for neigh in k_distance_neig[i]:
        lrd_sum += lrd[neigh[1]]
        rdist_sum += max(k_distance[neigh[1]][0], neigh[0])
    lof_list.append((i, lrd_sum * rdist_sum))
```


## 第六章 机器学习（I）

### **（1）为建模准备数据**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris()['data'], load_iris()['target']
# 将 Features 和 Target 合并
input_data = np.column_stack([X, y])
# 洗乱数据
data = np.random.shuffle(input_data)

# 分集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 检验训练集和测试集是否存在类别标签分布不平衡
def get_class_distribution(y):
    distribution = {}
    for label in set(y):
        distribution[label] = len(np.where(y==label)[0])
    distribution_pct = {
        label: count/sum(distribution.values()) 
        for label, count in distribution.items()
    }
    return distribution_pct

train_distribution = get_class_distribution(y_train)
test_distribution = get_class_distribution(y_test)
print("\nTrain dataset class label distribution\n" + '='*66)
for k,v in train_distribution.items():
    print(f"class label: {k:d}, percentage: {v:0.2f}")
print("\nTest dataset class label distribution\n" + '=' * 66)
for k,v in test_distribution.items():
    print(f"class label: {k:d}, percentage: {v:0.2f}")
```

### **（2）最邻近算法**

**衡量分类结果好坏的指标：**
- 混淆矩阵：label的真实值和预测值的排列矩阵，可以用 pandas 层次化DataFrame表示：

```python
confusion_matrix = pd.DataFrame(
    [['TP','FN'], ['FP','TN']],
    columns=pd.MultiIndex.from_tuples(
        [('Prediction','T'),('Prediction','F')]),
    index=pd.MultiIndex.from_tuples(
        [('Ground Truth','T'),('Ground Truth','F')])
    )
```
    上面的 columns 和index分别代表真实值和预测值的label，四种取值分别为以下的缩写：
        · True Positive -- 真正类
        · False Negative -- 漏报
        · False Positive -- 误报
        · True Negative -- 真负类
- 准确度：`Accuracy = Correct_Prediction / Total_Prediction`；其中：`Correct_Prediction = TP + TN`
- 另外还有错误率等其他指标

**K Nearest Neighborhood, KNN**
- 机械分类算法：把所有的训练数据加载到内存，当需要预测一个未知的实例时，在内存里比对所有的训练实例，匹配每一个属性来确定分类标签
- 上述算法中，如果找不到完全匹配的实例，就无法分配标签。KNN 可以理解为它的改进，不进行完全匹配，而是采用相似度量。
- 具体来说，KNN 在预测时，计算待预测实例与所有训练数据的距离，选择 K 个最近的实例，基于这K个最近邻的主体分类，对未知实例进行预测

```python
import itertools
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

def show_data(X, y):
    '''可视化数据集'''
    # 生成迭代器：[0,1,2,3] 的二元组合数
    col_pairs = itertools.combinations(range(4), 2)
    subplots = 231
    plt.figure(figsize=(8,9), dpi=80)#, facecolor='#999999')
    # 绘制每个特征两两组合的散点图
    for pair in col_pairs:
        plt.subplot(subplots)
        plt.scatter(X[:,pair[0]], X[:,pair[1]], c=y)
        plt.title(str(pair[0]) + '--' + str(pair[1]))
        plt.xlabel(str(pair[0])); plt.ylabel(str(pair[1]))
        subplots += 1
    plt.show()

def split_data(x, y):
    '''prepare a stratified train and test split'''
    sss = StratifiedShuffleSplit(test_size=0.2, n_splits=1)
    # 默认n_splits=10，也就是下面for循环会执行10次，但只保存了最后一次的值
    # 必须使用for循环，因为 StratifiedShuffleSplit 是惰性的生成器
    for train_idx, test_idx in sss.split(x, y):
        train_x = X[train_idx]
        train_y = y[train_idx]
        test_x = X[test_idx]
        test_y = y[test_idx]
    return train_x, train_y, test_x, test_y

# if __name__ == "__main__":
# 生成虚拟分类数据并可视化
X, y = make_classification(n_features=4)
show_data(X, y)    # 可见各个变量之间都存在多重线性关系
```

<img src="KNN-classification-dataset.jpg">

```python
# 生成分层的训练集和测试集，使训练集和测试集的标签分布一致
x_train, y_train, x_test, y_test = split_data(X, y)

# build and fit model
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
# 查看模型参数
print(knn.get_params())

# predict and evaluate model
print("\nModel evaluation on training set\n" + '='*66)
y_train_pred = knn.predict(x_train)
print(classification_report(y_train, y_train_pred))

print("\nModel evaluation on test set\n" + '='*66)
y_test_pred = knn.predict(x_test)
print(classification_report(y_test, y_test_pred))
```

### **（3）朴素贝叶斯分类**

- 贝叶斯公式：`P(X|Y) = P(Y|X) * P(X) / P(Y)`，即已知事件 Y 发生时，事件 X 发生的条件概率

- nltk：python 自然语言处理库

暂时不做这个...

### **（4）构建决策树解决多分类问题**

- 关于决策树的基础知识参考“R 统计学习笔记”
- 一些提高效率，短时间生成较合理的决策树的算法：
    - Hunt
    - ID3
    - C4.5
    - CART
- target 的属性种类：
    - 二元属性
    - 标称属性（n 个值）
    - 序数属性（如：小中大）
    - 连续属性（连续变量离散化形成）
- 决策树的不足：
    - 容易过拟合
    - 给定一个数据集，能产生巨量的决策树
    - 对类别不平衡敏感

```python
from pprint import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def split_data(X, y):
    sss = StratifiedShuffleSplit(test_size=0.2, n_splits=1)
    for train_idx, test_idx in sss.split(X, y):
        x_train = X[train_idx]; x_test = X[test_idx]
        y_train = y[train_idx]; y_test = y[test_idx]
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_iris()['data'], load_iris()['target']
    label_names = load_iris()['target_names']
    x_train, x_test, y_train, y_test = split_data(X, y)

    dtree = DecisionTreeClassifier(criterion='entropy')    #熵判断准则
    dtree.fit(x_train, y_train)

    # evaluation
    y_pred = dtree.predict(x_test)
    print("Model accuracy = {:0.4f}".format(accuracy_score(y_test, y_pred)))
    print("\nConfusion Matrix:\n" + '='*66)
    print(pprint(confusion_matrix(y_test, y_pred)))
    print("\nClassification Report:\n" + '='*66)
    print(classification_report(y_test, y_pred, target_names=label_names))
    # 导出树图（.dot 文件可以安装 graphviz 后使用 gvedit.exe 打开）
    export_graphviz(dtree, out_file='tree.dot')
```

<img src="decision-tree-dot-file.jpg">


## 第七章 机器学习（II）

### **（1）回归方法预测实数值**

```python
# coding = utf-8
""" 一个简单的OLS线性回归例子，使用Boston数据集。
"""
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def split_data(x, y, train_size, test_size, seed=1):
    x_train, x_temp, y_train, y_temp = \
        train_test_split(x, y, test_size=1-train_size, random_state=seed)
    size = test_size / (1 - train_size)
    x_dev, x_test, y_dev, y_test = \
        train_test_split(x_temp, y_temp, test_size=size, random_state=seed)
    return x_train, x_dev, x_test, y_train, y_dev, y_test

def plot_residual(model, x_train, y_train):
    prediction = model.predict(x_train)
    plt.figure(figsize=(8,9), dpi=80)
    plt.title("Residual Plot")
    plt.plot(prediction, y_train - prediction, 'go')
    plt.xlabel("Prediction"); plt.ylabel("Residual")
    plt.show()

def view_model(model, y, y_hat):
    print("\nModel Coefficients:\n" + '='*66)
    for i, coef in enumerate(model.coef_):
        print(f"\tCoefficient_{i+1:d}: {coef:0.2f}")
    print(f"\tIntercept: {model.intercept_:0.3f}")
    print("Mean Squared Error: {:0.2f}".format(mean_squared_error(y, y_hat)))

if __name__ == "__main__":
    '''最小二乘线性回归'''
    X, y = load_boston()['data'], load_boston()['target']
    x_train, x_dev, x_test, y_train, y_dev, y_test = split_data(X, y, 1/3, 1/3)
    lm = LinearRegression(normalize=True, fit_intercept=True)
    lm.fit(x_train, y_train)
    # 绘制残差
    plot_residual(lm, x_train, y_train)
    # 查看模型参数
    view_model(lm, y_train, lm.predict(x_train))
    # 验证集MSE
    print(mean_squared_error(y_dev, lm.predict(x_dev)))

    '''使用多项式特征'''
    polyfeat = PolynomialFeatures(degree=2, interaction_only=False)
    polyfeat.fit(x_train)
    x_train_poly = polyfeat.transform(x_train)
    x_dev_poly = polyfeat.transform(x_dev)
    poly_lm = LinearRegression(normalize=True, fit_intercept=True)
    poly_lm.fit(x_train_poly, y_train)
    plot_residual(poly_lm, x_train_poly, y_train)
    view_model(poly_lm, y_train, poly_lm.predict(x_train_poly))
    print(mean_squared_error(y_dev, poly_lm.predict(x_dev_poly)))

    '''递归特征选择方法--线性回归的低偏差高反差问题，也就是测试集误差大。'''
    # generate polynomial features
    polyfeat = PolynomialFeatures(interaction_only=True)
    polyfeat.fit(x_train)
    x_train_poly = polyfeat.transform(x_train)
    x_dev_poly = polyfeat.transform(x_dev)
    # build model
    lm = LinearRegression(normalize=True, fit_intercept=True)
    rfe_lm = RFE(estimator=lm, n_features_to_select=20)
    rfe_lm.fit(x_train_poly, y_train)
    # evaluate model
    plot_residual(rfe_lm, x_train_poly, y_train)
```

### **（2）岭回归**

```python
# coding = utf-8
""" 主要功能：在OLS回归中加入惩罚项，以限制权重参数的大小；详细参考ISLR。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def split_data(x, y, train_size, test_size, seed=1):
    x_train, x_temp, y_train, y_temp = \
        train_test_split(x, y, test_size=1-train_size, random_state=seed)
    size = test_size / (1 - train_size)
    x_dev, x_test, y_dev, y_test = \
        train_test_split(x_temp, y_temp, test_size=size, random_state=seed)
    return x_train, x_dev, x_test, y_train, y_dev, y_test


if __name__ == "__main__":
    # load data
    X, y = load_boston()['data'], load_boston()['target']
    X = X - np.mean(X, axis=0)
    x_train, x_dev, x_test, y_train, y_dev, y_test = split_data(X, y, 2/3, 1/9)

    # generate polynomial features
    polyfeat = PolynomialFeatures(interaction_only=True)
    polyfeat.fit(x_train)
    for name in ['train', 'dev', 'test']:
        exec(f"x_{name}_poly = polyfeat.transform(x_{name})")

    # build, fit and evaluate model
    rm = Ridge(normalize=True, alpha=0.015)
    rm.fit(x_train_poly, y_train)
    y_pred = rm.predict(x_train_poly)
    mse = mean_squared_error(y_train, y_pred)
    print("\nModel coefficients: \n" + '='*66)
    for i, coef in enumerate(rm.coef_):
        print(f"\tCoefficient_{i+1:d}:\t{coef:0.3f}")
    print(f"Intercept: {rm.intercept_}")

    # repeat above on dev and test split ...

    # 向X中加入噪音，测试模型敏感性（非常敏感）
    # X = X + np.random.normal(0, 1, (X.shape))

    # 参数查找
    alphas = np.linspace(10, 100, 300)    # [10 - 100], length=300
    coefficients = []
    for a in alphas:
        model = Ridge(normalize=True, alpha=a)
        model.fit(X, y)
        coefficients.append(model.coef_)
    # 绘制每个参数（共13个）随着alpha的变化
    plt.figure(figsize=(6,6.75), dpi=80)
    plt.title("Coefficient Weights for Different Alpha Values")
    plt.plot(alphas, coefficients)
    plt.xlabel("Alpha"); plt.ylabel("Weight")
    plt.show()
```

### **（3）lasso**

```python
# coding = utf-8
""" 与岭回归相比，Lasso可以筛选参数，具有稀疏特性，详细参考ISLR。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression

# load data
X, y = load_boston()['data'], load_boston()['target']

# build models on parameters grid and get coefficients
alphas = np.linspace(0, 0.5, 200)
model = Lasso(normalize=True)
coefficients = []
for alpha in alphas:
    model.set_params(alpha=alpha)
    model.fit(X, y)
    coefficients.append(model.coef_)

# 绘制权重随着参数的变化
plt.figure(figsize=(6,6.75), dpi=80)
plt.title("Coefficient Weights for Different Alpha Values")
plt.plot(alphas, coefficients)
plt.vlines(x=0.1, ymin=-2, ymax=5, color='r', alpha=0.1)
plt.xlabel("Alpha"); plt.ylabel("Weight")
plt.axis('tight')
plt.show()

# 根据上面的图示选择合适参数，暂定选择alpha=0.1，保留4个变量
model = Lasso(normalize=True, alpha=0.1)
model.fit(X, y)
# 也可以使用mse作为criterion递归查找最优参数

# 模型参数和MSE（训练集误差）
print("\nModel Coefficients: \n" + '='*66)
for i, coef in enumerate(model.coef_):
    print(f"\tCoefficient_{i+1:d}:\t{coef:0.3f}")
print(f"Intercept: {model.intercept_}")
print("MSE: {:0.3f}".format(mean_squared_error(y, model.predict(X))))

```

### **（4）L1 缩减和 L2 缩减交叉验证**

```python
# coding = utf-8
""" ·现实场景中，数据集一般不够大，可以通过交叉验证进行模型选择。详细参考ISLR。
    ·交叉验证迭代器的使用参考documentation的examples
    ·以下对lasso回归模型进行交叉验证。
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

# load and split data
X, y = load_boston()['data'], load_boston()['target']
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.3, random_state=0)
# generate polynomial features
polyfeat = PolynomialFeatures(interaction_only=True)
polyfeat.fit(x_train)
for name in ['train', 'test']:
    exec(f"x_{name}_poly = polyfeat.transform(x_{name})")

# build model
rr = Ridge(normalize=True)
kfold = KFold(n_splits=5)
param_grid = {'alpha': np.linspace(0.0015, 0.0017, 30)}
grid = GridSearchCV(estimator=rr,
                    param_grid=param_grid,
                    cv=kfold,
                    scoring='neg_mean_squared_error')
# 调用fit实例方法，将在定义的参数范围内，进行k折交叉验证
grid.fit(x_train_poly, y_train)

# 查看不同参数的结果
print(grid.cv_results_)
print(grid.best_params_)
# 保存最佳模型
best_model = grid.best_estimator_
print(best_model.coef_)
# 计算最佳模型的测试集MSE
mse = mean_squared_error(y_train, best_model.predict(x_train_poly))

'''
关于GridSearchCV对象的更多属性和方法以及其他交叉验证器请参考scikit-learn文档。
'''
```


## 第八章 集成方法


集成学习（Ensemble Learning）的概念：不仅仅通过个人，而是通过集体智慧来做出决策。准确来说，就是生成大量模型，用它们来进行预测。从多个相近的模型输出的结果会比仅仅从一个模型的到的结果更好。集成的模型可以是不同类别的，比如同时利用神经网络模型和贝叶斯模型。本章只讨论集成同类的模型。

挂袋法，每个模型只使用训练集的一部分，以减少过拟合。挂袋法很容易实现并行化，可以同时处理的不同的训练集样本。挂袋法对如线性回归之类的线性预测器无效。

提升法，产生一个逐步复杂的模型序列，基于上一个模型的错误，训练新的模型。每次训练的模型被赋予一个权重，并按权重得出最终的预测值。可见提升法是按顺序执行的，无法并行化。提升法常常用一些弱分类器，如单层决策树。

### **（1）装袋法**

集成方法属于基于评委学习方法一族。挂袋法也叫引导聚集，只有在潜在的模型能产生不同变化时才有效。也就是要产生有轻微变化的多种模型。

使用自举方法在模型上产生变化，自举就是在数据集中随机采样一定的观测，不管是否有替换。在挂袋法中，通过自举产生 m 个数据集，然后对每一个建立一个模型，最终使用所有模型的输出来产生最终的预测。对于回归问题就是取均值，对于分类问题使用投票方法。

挂袋法适用于不稳定的、对变化敏感的模型，例如决策树，尤其未剪枝的。而不适合 KNN 等稳定的分类器。应用于 KNN 时需要使用随机子空间方法（例如，在集成的每个模型里随机选择特征属性的子集）。

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

# generate a dataset for classification
X, y = make_classification(n_samples=500, n_features=30, flip_y=0.03,
    n_informative=18, n_redundant=3, n_repeated=3, random_state=1)
# 一共30个特征，18个关键特征，3个多余的特征，3个重复的特征，flip_y使分类更困难

# split data
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)

# generate a KNN classifier, then report
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_train)
print(classification_report(y_train, y_pred_knn))

# generate a Bagging classifier on KNN, then report
bagging = BaggingClassifier(KNeighborsClassifier(), n_estimators=100,
                            max_samples=1.0, max_features=0.7,
                            bootstrap=True, bootstrap_features=True,
                            random_state=1)
# estimator自举时选用100%的样本，选取70%的特征属性（随机空间方法）
# 样本与特征的采样都是有放回（default=False）
bagging.fit(x_train, y_train)
y_pred_bagging = bagging.predict(x_train)
print(classification_report(y_train, y_pred_bagging))
# 可以看到bagging比knn的准确率明显提高

# 打印出前 3 个模型所选取的特征属性
print("\nSampled attributes in top 3 estimators: \n" + '='*66)
for i, features in enumerate(bagging.estimators_features_[0:3]):
    print(f"estimator_{i+1:d}: ", features)

# 在验证集上对比knn和bagging
print("\nSingle KNN: \n" + '='*66)
print(classification_report(y_dev, knn.predict(x_dev)))
print("\nBagging KNN: \n" + '='*66)
print(classification_report(y_dev, bagging.predict(x_dev)))
```

### **（2）提升法**

**a) Bootstrap 原理步骤：**

二分类问题，分类器的输入可以表达为：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`X = {x1, x2, ..., xN} and Y = {0, 1}`

分类器的任务就是找到一个可以近似的函数：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Y = F(X)`

分类器的错误率定义为：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`error rate = 1/N * sum([instance where yi != F(xi)])`

假设构建一个弱分类器（错误率仅好于随机猜测），然后通过提升法构建一系列弱分类器用在经过微调的数据集上，每个分类器使用的数据只做了很小的调整。最后结束于第 M 个分类器：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`F1(X), F2(X), ..., FM(X)`

最后把各个分类器生成的预测集成起来，进行加权投票：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`F_final(X) = sign( sum(alpha_i * F_i(X)) )`

**b) 模型权重计算原理步骤：**

首先修改错误率公式：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`error_rate = sum( wi * abs(yi - yi_pred) ) / sum( wi )`

对于 N 个实例，每个实例的权重为 1/N ，wi 表示模型的初始权重 `wi = n / N` ，n 为采样数目。

根据调整的 error_rate 计算每个模型的 alpha：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`alpha_i = 1/2 * log( (1 - error_rate + epsilon) / (error_rate + epsilon) )`

其中epsilon是一个微小的值。

最终计算每个模型的输出权重：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`wi = wi * exp( alpha_i * abs(yi - yi_pred) )`

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# generate dataset for classification
X, y = make_classification(n_samples=500, n_features=30, flip_y=0.03,
                           n_informative=18, n_redundant=3, n_repeated=3,
                           random_state=1)
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)

# build decision tree classifier and report
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_train)
print("\nSingle Model Accuracy on training data:\n" + '='*66)
print(classification_report(y_train, y_pred_dtc))
print("Fraction of Misclassification: {:0.2f}%".format(
                                    zero_one_loss(y_train, y_pred_dtc)*100))

# build boosting classifier and report
boosting = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
            n_estimators=85,
            algorithm='SAMME',
            random_state=1)
# 决策树只有树桩，叶子结点最小样本数为1
# SAMME：AdaBoost的增强版：
#   Stage wise Additive Modeling using Multi-class Exponential loss function

boosting.fit(x_train, y_train)
y_pred_boost = boosting.predict(x_train)
print("\nBoosting Model Accuracy on training data:\n" + '='*66)
print(classification_report(y_train, y_pred_boost))
print("Fraction of Misclassification: {:0.2f}%".format(
                                    zero_one_loss(y_train, y_pred_boost)*100))

# boosting model 参数
print("\nEstimator Weights and Error on Boosting Models: " + '=' * 66)
for i, weight in enumerate(boosting.estimator_weights_):
    e = boosting.estimator_errors_[i]
    print(f"estimator_{i + 1:d}: \tweight = {weight:0.4f}\t\terror = {e:0.4f}")
```

```python
'''绘制模型数量与错误率的关系，同时对比single decision tree 模型'''
error_rates = []
error_rates_dev = []
dtc_error_rates = []
dtc_error_rates_dev = []
numbers = [i for i in range(20, 120, 10)]

for e in numbers:
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    boosting = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1,min_samples_leaf=1),
            n_estimators=int(e), algorithm='SAMME', random_state=1 )
    boosting.fit(x_train, y_train)
    y_pred = boosting.predict(x_train)
    y_dev_pred = boosting.predict(x_dev)
    y_pred_dtc = dtc.predict(x_train)
    y_dev_pred_dtc = dtc.predict(x_dev)
    error_rates.append(zero_one_loss(y_train, y_pred))
    error_rates_dev.append(zero_one_loss(y_dev, y_dev_pred))
    dtc_error_rates.append(zero_one_loss(y_train, y_pred_dtc))
    dtc_error_rates_dev.append(zero_one_loss(y_dev, y_dev_pred_dtc))

plt.figure(figsize=(6, 6.75), dpi=80)
plt.title("Number of Estimators VS. Mis-classification Rate\n" +
                "--compared with single decision tree model")
plt.plot(numbers, error_rates, label="Train--AdaBoost")
plt.plot(numbers, error_rates_dev, label="Dev--AdaBoost")
plt.plot(numbers, dtc_error_rates, label="Train--DTC")
plt.plot(numbers, dtc_error_rates_dev, label="Dev--DTC")
plt.xlabel('Number of Estimators'); plt.ylabel('Mis-classification Rate')
plt.legend(loc='upper right')
plt.show()
''' 结果显示：
1.boosting模型测试集错误率明显低于弱分类器dtc；但是
2.模型数量越大训练集误差越小，但是测试集误差只能有限减小，最后震荡在一个范围。
'''
```

<img src="bootstrap-result-compare.png">

### **（3）梯度提升**

- 提升法：用一种渐进的，阶段改良的方式，从一系列若分类器适配出一个增强模型。具体就是通过错误率调整实例的权重，在下一模型改进不足之处。
- 梯度提升法就是采用梯度而不是权重来鉴别缺陷。以下是一个简单回归问题的梯度提升步骤。
    - 给定预测变量 X 和响应变量 Y：
        - `X = {x1, x2, ..., xN} and Y = {y1, y2, ..., yN}`
    - 先从简单模型开始，例如直接使用均值预测所有值：
        - `y_hat = sum(yi) / N`
    - 得到残差：
        - `Ri = yi - y_hat`
    - 下一个分类在以下数据上训练：
        - `{(x1, R1), (x2, R2), ..., (xN, RN)}`
    - 进行迭代达到所需准确率
- 为何叫做梯度提升：`F(xi) - yi` 就代表梯度，即改点的一阶导数，正好是负的残差
- 梯度提升是一种框架而不仅仅是一种具体的算法，任何可微函数都可以应用到框架中

```python
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

def view_model(model):
    print("\nTraining Scores: \n" + '='*66)
    for i, score in enumerate(model.train_score_):
        print(f"Estimator_{i+1:d} Score:\t{score:0.3f}")
    print("\nFeature Importance: \n" + '='*66)
    for i, score in enumerate(model.feature_importances_):
        print(f"Feature_{i+1} Importance:\t{score:0.3f}")
    plt.figure(figsize=(6, 6.75), dpi=80)
    plt.plot(range(1, model.estimators_.shape[0]+1), model.train_score_)
    plt.xlabel("Model Sequence"); plt.ylabel("Model Score")
    plt.show()

# load boston dataset
X, y = load_boston()['data'], load_boston()['target']
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)

# generate polynomial features
polyfeat = PolynomialFeatures(2, interaction_only=True)
polyfeat.fit(x_train)
for name in ['train', 'dev', 'test']:
    exec(f"x_{name}_poly = polyfeat.transform(x_{name})")

# build gradient boosting model and report
poly_gbr = GradientBoostingRegressor(n_estimators=500, verbose=10,
            subsample=0.7, learning_rate=0.15, max_depth=3, random_state=1)
# 当verbose>1，将每个模型或树构建时的进展情况打印出来；
# 采样率subsample指定模型采样70%的训练集数据；
# 学习率learning_rate，是一个缩减参数，控制每个模型的贡献；

poly_gbr.fit(x_train_poly, y_train)
y_pred = poly_gbr.predict(x_train_poly)
print("\nModel Performance in Training Set: \n" + '='*66)
view_model(poly_gbr)
print("\nMSE:\t{:0.2f}".format(mean_squared_error(y_train, y_pred)))

# 在验证集上测试结果
y_dev_pred = poly_gbr.predict(x_dev_poly)
print("\nModel Performance in Dev Set: \n" + '='*66)
print("\nMSE:\t{:0.2f}".format(mean_squared_error(y_dev, y_dev_pred)))
```


## 第九章 生长树

上述基于树算法具备对抗噪声的健壮性和解决各类问题的广泛能力，能在没有数据整理的情况下获得很好的结果，可作为黑箱工具。

主要优点：挂袋法有天然的并行性；决策树算法在每一层将数据划分，实现隐式的特征选择；几乎不需要对数据进行预处理，不同度量、缺失值或者异常值，以及非线性问题等，对其几乎没有影响。

主要难题：为了避免过拟合进行剪枝的难度；大型树容易导致低偏差和高方差。

### **（1）随机森林**

- 随机森林也是一种挂袋法，基本思路是利用大量的噪声评估器，用平均法处理噪声，以减小最终结果的方差。
- 随机森林构建的是相互之间没有关联的树。具体方法是，在进行结点划分时，不选择所有的属性，而是随机选择一个属性子集。
- 构建随机森林：LOOP for 1 to T
    - 随机选择 m 个属性
    - 采用预定义的 criterion，选择一个最佳属性作为划分变量
    - 将数据集划分为两个部分
    - 返回划分的数据集，分别在两个部分迭代上述过程
    - 最终获得 T 棵树
- 提升法使用弱分类器进行强化，实现较低的方差（较高的验证集准确率）；而在随机森林中，构建最大深度的树，但是引入了高方差。随后通过大量树进行投票，解决高方差问题。

```python
from operator import itemgetter
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# generate dataset for classification
X, y = make_classification(n_samples=500, n_features=30, flip_y=0.03,
                           n_informative=18, n_redundant=3, n_repeated=3,
                           random_state=1)
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)

# build forest and report
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_train)
y_dev_pred = rfc.predict(x_dev)
train_score = accuracy_score(y_train, y_pred)
dev_score = accuracy_score(y_dev, y_dev_pred)
print(f"Training Accuracy: {train_score:0.2f}\tDev Accuracy: {dev_score:0.2f}")
# 结果方差较大，训练集准确率100%，测试集只有83%

# search parameters
param_dict = {
    "n_estimators": np.random.randint(75, 200, 20),
    "criterion": ['gini', 'entropy'],   #基尼系数与熵判据
    "max_features": [int(np.sqrt(X.shape[1]))*i for i in [1,2,3]] + \
                    [int(np.sqrt(X.shape[1]))+10]   #每个结点选取的特征数
    }
grid = RandomizedSearchCV(estimator=RandomForestClassifier(),
                          param_distributions=param_dict,
                          n_iter=20, random_state=1, n_jobs=-1, cv=5)
grid.fit(X, y)

# view parameters of grid searching
print("\n模型参数：\n" + '='*66)
for score in grid.cv_results_['params']:
    print(score)
print(classification_report(y_dev, grid.predict(x_dev)))
# 当对grid使用predict实例方法时，隐式调用了最优模型
best_model = grid.best_estimator_
print(best_model)
```

### **（2）超随机树**

- 超随机树（Extra Trees）或称为极限随机树（Extremely Randomized Trees）与随机森林主要有两点不同：
    - 不使用自举法，而是使用完整的数据集
    - 给定结点随机选择属性数量 K，它随机选择割点，不考虑目标变量
- 优势：更好地降低方差，可以在未知数据集上取得很好的效果；并且计算复杂度相对较低
- 构建超随机树：LOOP for 1 to T
    - 随机选择 m 个属性
    - 随机选取一个属性作为划分变量（忽略任何标准，完全随机）
    - ... 接下来与随机森林一致

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# generate dataset for classification
X, y = make_classification(n_samples=500, n_features=30, flip_y=0.03,
                           n_informative=18, n_redundant=3, n_repeated=3,
                           random_state=1)
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)

# build forest and report
etc = ExtraTreesClassifier(n_estimators=100, random_state=1)
etc.fit(x_train, y_train)
y_pred = etc.predict(x_train)
y_dev_pred = etc.predict(x_dev)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred):0.2f}")
print(f"Dev Accuracy: {accuracy_score(y_dev, y_dev_pred):0.2f}")
print("\nCross Validated Score: \n" + '='*66)
print(cross_val_score(etc, x_dev, y_dev, cv=5))

# search parameters (与上一节一样流程，只estimator由随机森林变成extra_trees)
param_dict = {
    "n_estimators": np.random.randint(75, 200, 20),
    "criterion": ['gini', 'entropy'],   #理论来说不需要判断准则
    "max_features": [int(np.sqrt(X.shape[1]))*i for i in [1,2,3]] + \
                    [int(np.sqrt(X.shape[1]))+10]   #每个结点选取的特征数
    }
grid = RandomizedSearchCV(estimator=ExtraTreesClassifier(),
                          param_distributions=param_dict,
                          n_iter=20, random_state=1, n_jobs=-1, cv=5)
grid.fit(X, y)
print("\n模型参数：\n" + '='*66)
for score in grid.cv_results_['params']:
    print(score)
print(classification_report(y_dev, grid.predict(x_dev)))
# 当对grid使用predict实例方法时，隐式调用了最优模型

best_model = grid.best_estimator_
print(best_model)
# 在验证集的准确率达到100%
```

### **（3）旋转森林**

- 随机森林和挂袋法使用大量的树实现精确的预测，而旋转森林的思路是使用少得的集成数量。
- 构建旋转森林步骤：LOOP for 1 to T
    - 将训练集的属性划分为大小相等的 K 个不重叠子集
    - 对每个子集，自举 75% 的样本，在样本上执行：
        - 在 K 个数据集的第 i 个子集中进行 PCA，保留主成分。对每个特征 j， 主成分标为 a_ij
        - 保留以上所有子集的主成分
    - 创建 `n*n` 的旋转矩阵，n 是属性总数
    - 将上述的主成分放进旋转矩阵，这些成分匹配特征在初始训练数据集中的位置
    - 通过矩阵乘法将训练集投影到旋转矩阵上
    - 用投影的数据构建一颗决策树，并保存树和旋转矩阵

```python
"""python 中还没有包含旋转矩阵算法的库（2016-12）需要自己编写"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

# generate dataset for classification
X, y = make_classification(n_samples=500, n_features=50, flip_y=0.03,
                           n_informative=30, n_redundant=5, n_repeated=5,
                           random_state=1)
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)

def random_subset(iterable_obj, k):
    '''get random subset from iterable object, using in next function'''
    subsets = []
    iteration = 0
    limit = len(iterable_obj) / k  #先储存limit，iterable_obj的长度会动态变化
    # 洗牌：
    np.random.shuffle(iterable_obj)
    # 获取k个子集：
    while iteration < limit:
        if k <= len(iterable_obj):
            subset = k
        else:
            subset = len(iterable_obj)
        subsets.append(iterable_obj[:subset])
        del iterable_obj[:subset]    #删除该切片，使子集不相交
        iteration += 1
    return subsets

# ========================== 生成旋转森林的主循环 ===========================

T = 25    #树的数量T
K = 5    #特征子集长度K
models = []    #保存每一棵树
r_matrices = []    #保存与树对应的旋转矩阵
feature_subsets = []    #保存随机选取的特征子集

for i in range(T):
    # 训练集自举，随机选取75%的子集：
    x, _, _, _ = train_test_split(x_train, y_train, test_size=0.25)
    # 在数据集上随机选取特征（生成用于选集的index）：
    feature_idx = [idx for idx in range(x.shape[1])]
    random_k_subset = random_subset(feature_idx, K)
    feature_subsets.append(random_k_subset)

    # 生成旋转矩阵：
    rotation_matrix = np.zeros((x.shape[1],x.shape[1]), dtype=float)  #初始化
    for subset in random_k_subset:
        # 对每个特征子集执行PCA，并将所有components存放进旋转矩阵的对应位置：
        # 注意到这个循环只填满一个旋转矩阵，因为每一个subset之间是互斥的
        pca = PCA()
        x_subset = x[:, subset]
        pca.fit(x_subset)
        for ii in range(len(pca.components_)):
            for jj in range(len(pca.components_)):
                rotation_matrix[subset[ii],subset[jj]] = pca.components_[ii,jj]
    r_matrices.append(rotation_matrix)

    # 将输入数据投影到旋转矩阵
    x_transformed = x_train.dot(rotation_matrix)
    # 使用转换的数据构建决策树并保存
    dtc = DecisionTreeClassifier()
    dtc.fit(x_transformed, y_train)
    models.append(dtc)

# =========================== 评估旋转森林模型 =============================
def predict_models(x, models=models, r_matrices=r_matrices):
    '''采用全部模型的预测值进行投票产生最终预测'''
    # 获取每个模型的预测值
    y_predictions = []
    for i, model in enumerate(models):
        x_trans = x.dot(r_matrices[i])
        y_pred = model.predict(x_trans)
        y_predictions.append(y_pred)
    # 结合所有模型的预测得出最终预测
    prediction_matrix = np.matrix(y_predictions)
    final_predictions = []
    for i in range(x.shape[0]):
        predictions = np.ravel(prediction_matrix[:,i])    #降维，扁平化
        nonzero_idx = np.nonzero(predictions)[0]    #返回非零的坐标(元组里面)
        is_one = len(nonzero_idx) > len(models)/2    #投票，过半数为1则返回True
        final_predictions.append(is_one)
    return np.array(final_predictions)

# 分别在训练集和验证集上评估
y_pred = predict_models(x_train)
y_dev_pred = predict_models(x_dev)
print(classification_report(y_train, y_pred))
print(classification_report(y_dev, y_dev_pred))

# 跑了几次，发现效果不好，方差较大。
```


## 第十章 大规模机器学习 -- 在线学习

本章的关注点是大规模机器学习以及适合处理大规模问题的算法。前几章的所有模型的数据集都是可以完全加载进计算机的内存的。当数据集数目庞大，需要构建一个框架，可以根据部分数据进行判断，并随着新数据的输入而持续提高。

随机梯度下降（Stochastic Gradient Descent）就是一个这样的框架。许多线性方法，如逻辑斯蒂回归、线性回归、线性 SVM，以及非线性的核方法等都可用于 SGD 框架。

### **（1）用感知器作为在线学习算法**

- 感知器（Perceptron）：处理大规模学习问题，建模时一次只使用数据集的一部分
- 具体算法：
    - 将模型权重用一个小的随机数初始化
    - 用输入数据 X 的均值进行去中心化
    - 在每个步骤 t 中（或称为纪元）：
        - 随机选择记录中的一个实例进行预测
        - 比较预测标签和真实标签的误差
        - 如果预测错误则更新权重
- 如何更新权重？
    - 假定在一个纪元中，输入 X 为：
        - `Xi = {x1, x2, ..., xm},    i=1, 2, ..., n`
    - Y 的集合为：
        - `Y = {+1, -1}`
    - 则权重定义为：
        - `W = {w1, w2, ..., wm}`
    - 每条记录得出的预测值为：
        - `yi_hat = sign(wi * xi)`
    - 权重的更新公式为：
        - `w_t + 1 = w_t + yi * xi`
    - 增加学习速率参数 alpha：
        - `w_t + 1 = w_t + alpha * yi * xi`
        - alpha 取值一般为 [0.1, 0.4]

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, PolynomialFeatures

def get_data(batch_size, kernel='linear'):
    '''生成多组数据模拟大型数据和数据流，隐式返回一个生成器generator'''
    b_size = 0
    while b_size < batch_size:
        X, y = make_classification(n_samples=1000, n_features=30, flip_y=0.03,
                                   n_informative=24, n_redundant=3,n_repeated=3,
                                   random_state=1)
        # 添加非线性核选项：
        if kernel == 'linear':
            X = scale(X, with_mean=True, with_std=True)    #特征数据中心化
        elif kernel == 'polynomial':
            polyfeat = PolynomialFeatures(degree=2)
            X = polyfeat.fit_transform(X)
        y[y<1] = -1    #把y标签由{0, 1}改为{-1, +1}
        b_size += 1
        yield X, y

def build_perceptron(x, y, weights, epochs, alpha=0.5):
    ''' 构建一个感知器模型，用于对于多组数据进行循环建模，模拟现实的数据流。
        weights: 初始权重矩阵；
        epochs: 纪元数，即更新权重的次数；
        alpha: 学习速率。
    '''
    for i in range(epochs):
        # 搅乱数据集：
        shuffled_idx = [i for i in range(len(y))]
        np.random.shuffle(shuffled_idx)    #这个函数是就地更改的
        x_train = x[shuffled_idx, :].reshape(x.shape)
        y_train = np.ravel(y[shuffled_idx])

        # 一次构建一个实例的权重：
        for idx in range(len(y)):
            prediction = np.sign(np.sum(x_train[idx,:] * weights))
            if prediction != y_train[idx]:
                weights = weights + alpha * (y_train[idx] * x_train[idx, :])
    # 所有纪元更新完之后返回权重
    return weights

# 生成 10 组数据流
data = get_data(10, 'linear')    #data = get_data(10, 'polynomial')
# 使用生成器的__next__()方法获取下一组数据
X, y = data.__next__()

# 初始化权重矩阵
weights = np.zeros(X.shape[1])

# 模拟收到10组数据集，进行建模学习
for i in range(10):
    weights = build_perceptron(X, y, weights, 100, 0.5)
    y_pred = np.sign(np.sum(X * weights, axis=1))
    print('='*66+f"\nModel Performance After Receiving Dataset Batch {i+1}:\n")
    print(classification_report(y, y_pred))
    if i != 9:
        X, y = data.__next__()
```

### **（2）用梯度下降解决回归问题**

**标准的回归结构**中，有一系列实例：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`X = {x1, x2, ..., xn}`

每个实例有 m 个属性（特征）：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Xi = {xi1, xi2, ..., xim},    i = 1, 2, ..., m`

回归算法的任务是找到一个 X 到 Y 映射：`Y = F(X)` 。由一个权重向量来进行参数化：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`Y = F(X, W) = <X, W> + b`

于是回归问题就变成寻找最优权重的问题。采用损失函数进行优化，对 n 个实例的数据集，全局损失函数形式为：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`1/n * sum( L(f(xi, w), yi) )`

**随机梯度下降（Stochastic Gradient Descent, SGD）**是一种优化技术，可用于最小化损失函数。首先要找出 L(.) 的梯度，也就是损失函数对权重 w 的偏导

和批量梯度下降等其他技术不同，SGD 每次只操作一个实例：
- 对每个纪元 t ，搅乱数据集
- 选择一个实例 xi 及其对应的响应变量 y
- 计算损失函数及其对于 w 的偏导 nabla_w（倒三角符号，表示矢量求偏导）
- 更新权重值

更新权重值的公式为：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`w_t + 1 = w_t - nabla_w L(yi_hat, yi)`

式中权重和梯度的方向相反，这样迫使权重向量降序排列，以减小目标函数。引入学习率后表示为：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`w_t + 1 = w_t - eta * (nabla_w L(yi_hat, yi))`

在 SGD 的基础上加上正则化，类似岭回归增加 L2 范数正则项和学习率，权重更新公式表示为：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`w_t + 1 = w_t - eta * (nabla_w L(yi_hat, yi)) + alpha * (nabla_w R(W))`

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import SGDRegressor


# generate dataset for regression
X, y = make_regression(n_samples=1000, n_features=30, random_state=1)
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)


# 建模 & 预测
sgdr = SGDRegressor(n_iter=10, shuffle=True, loss='squared_loss',
                    learning_rate='constant', eta0=0.01, fit_intercept=True,
                    penalty='none')
# 上述大部分参数是默认的，写出来看一下而已
# learning_rate指定学习速率的类型，eta0指定学习速率的值
# penalty指定缩减类型（Lr缩减）,本例无需进行缩减
sgdr.fit(x_train, y_train)
y_pred = sgdr.predict(x_train)
y_dev_pred = sgdr.predict(x_dev)

# 查看模型参数
print(f"\nModel Intercept: {sgdr.intercept_}")
for i, coef in enumerate(sgdr.coef_):
    print(f"Coefficient_{i+1}: {coef:0.3f}")
# 打印训练集和验证的MSE和MAE
print('\nTrain set: ========>>')
print(f"Mean absolute error: {mean_absolute_error(y_train, y_pred):0.2f}")
print(f"Mean squared error: {mean_squared_error(y_train, y_pred):0.2f}")
print('\nDev set: ========>>')
print(f"Mean absolute error: {mean_absolute_error(y_dev, y_dev_pred):0.2f}")
print(f"Mean squared error: {mean_squared_error(y_dev, y_dev_pred):0.2f}")

# 使用 L2 正则化建模：通过penalty参数指定正则项
regularized_sgdr = SGDRegressor(n_iter=10, learning_rate='constant',
                            penalty='l2', alpha=0.01)    #alpha:正则项缩放
regularized_sgdr.fit(x_train, y_train)
# 查看模型参数
print("\nRegularized Model Coefficients: \n" + '='*66)
for i, coef in enumerate(regularized_sgdr.coef_):
    print(f"Coef_{i+1:d}:\t{coef:0.3f}")
print(f"Model Intercept:\t{regularized_sgdr.intercept_}")
```

### **（3）用梯度下降解决分类问题**

分类问题除响应变量不同，其他结构与回归问题类似。由于响应变量的性质，需要不一样的损失函数。对于二分类问题，应用逻辑斯蒂回归函数获取预测值：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`F(w, xi) = 1 / (1 + e^(-xi * w^T))`

上式被称为 sigmoid 函数。对于 xi 很大的正数，函数值趋向 1，反之趋向 0。于是可以定义如下的 log 损失函数：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`L(w, xi) = -yi * log(F(w, xi)) - (1-yi) * log(1 - F(w, xi))`

将上式代入 SGD 的权重更新公式，即得到分类问题的 SGD。

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


# generate dataset for classification
X, y = make_classification(n_samples=1000, n_features=30, flip_y=0.03,
                           n_informative=18, n_redundant=3, n_repeated=3,
                           random_state=1)
# split dataset
x_train, x_temp, y_train, y_temp = \
            train_test_split(X, y, test_size=0.3, random_state=1)
x_dev, x_test, y_dev, y_test = \
            train_test_split(x_temp, y_temp, test_size=0.3, random_state=1)


# build model and predict
sgdc = SGDClassifier(n_iter=50, loss='log',    #使用log损失函数
                     learning_rate='constant', eta0=1e-4, penalty='none')
sgdc.fit(x_train, y_train)
y_pred = sgdc.predict(x_train)
y_dev_pred = sgdc.predict(x_dev)

# evaluate model and show score
print("Training Accuracy: {0:0.2f}\tDev Accuracy: {1:0.2f}".format(\
        accuracy_score(y_train, y_pred), accuracy_score(y_dev, y_dev_pred)))
```

---------------------
**END**