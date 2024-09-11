# K-近邻算法（K-Nearest Neighbors, KNN）

## 0x01. KNN概念

K-近邻算法（K-Nearest Neighbors, 简称 KNN）是一种**基于实例的学习算法**，用于分类和回归问题。KNN 的核心思想是，通过计算样本之间的距离，将新样本归类到距离最近的 \( K \) 个邻居所属的类别中。

KNN 被称为懒惰学习算法，因为在训练阶段并没有显式构建模型，而是把所有训练数据存储起来，等到需要分类或预测时才进行计算。

## 0x02. KNN的工作原理

1. **选择距离度量方式**：通常使用**欧氏距离**来度量样本间的距离：
   $$ 
   d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} 
   $$
   其中 \( x_i \) 和 \( y_i \) 分别是样本 \( x \) 和样本 \( y \) 在第 \( i \) 个特征上的取值。

2. **确定最近的 \( K \) 个邻居**：计算待分类样本与训练集中所有样本之间的距离，选出距离最近的 \( K \) 个样本。

3. **分类或回归**：
   - **分类问题**：选择 \( K \) 个邻居中出现次数最多的类别作为新样本的类别。
   - **回归问题**：取 \( K \) 个邻居的目标值的平均值作为新样本的预测值。

## 0x03. KNN的参数

1. **K值**：K 值是 KNN 的关键参数，表示参与决策的邻居数量。选择合适的 \( K \) 值对于算法性能至关重要。
   - **小 K 值**：较小的 \( K \) 值会使模型对噪声数据更敏感，容易导致过拟合。
   - **大 K 值**：较大的 \( K \) 值会使模型更加平滑，但可能会忽略局部结构，导致欠拟合。

2. **距离度量方式**：
   - **欧氏距离（Euclidean Distance）**：常用于连续数据，公式如前所述。
   - **曼哈顿距离（Manhattan Distance）**：常用于离散特征：
     $$ 
     d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
     $$
   - **闵可夫斯基距离（Minkowski Distance）**：可以视为欧氏距离和曼哈顿距离的泛化形式：
     $$ 
     d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} 
     $$
     当 \( p = 2 \) 时为欧氏距离，\( p = 1 \) 时为曼哈顿距离。

3. **加权 KNN**：在 KNN 中可以给邻居的距离加权，距离越近的邻居权重越大，使得预测更加准确。


## 0x04. 优缺点

### 优点：
- **简单直观**：KNN 是一种非常直观的算法，简单易实现，不需要训练过程。
- **适应性强**：KNN 不依赖特定的数据分布，适用于分类和回归任务。

### 缺点：
- **计算量大**：KNN 在预测时需要计算所有训练样本的距离，计算复杂度较高，特别是在数据量较大的情况下。
- **高维数据问题**：KNN 在高维空间中会遭遇“维度灾难”问题，距离度量变得不再可靠。
- **对不平衡数据敏感**：在类别不平衡的数据集中，KNN 倾向于将样本归类到多数类。

## 0x05. KNN的应用场景

1. **文本分类**：KNN 可以用于自然语言处理中的文本分类任务。通过将文本转换为向量表示，计算不同文本之间的相似度，从而实现分类。
2. **推荐系统**：KNN 可以通过寻找相似用户或相似物品，来为用户提供推荐。
3. **图像分类**：KNN 也可以用于图像分类，尤其是基于低维特征表示的图像识别任务。

## 0x06. KNN的实现

下面是一个基于 Python 的 KNN 实现的代码示例：

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建KNN分类器，选择K=3
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率为: {accuracy}")


## 0x07. KNN优化
KD-树和球树：为了加速距离计算，可以使用 KD-树（k-d tree）或球树（Ball Tree）等数据结构来存储训练样本，加快 KNN 的查询效率。
标准化和归一化：在 KNN 中，距离度量对特征的量纲敏感，因此常常需要对数据进行标准化或归一化处理，以消除不同特征的量纲差异对结果的影响。
降维：对于高维数据，可以使用主成分分析（PCA）等降维方法来降低特征维度，减小计算量并提高性能。
