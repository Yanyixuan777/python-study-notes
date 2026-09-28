# IBM Exploratory Data Analysis for Machine Learning

## 全课程复习笔记

作者：yanyixuan

覆盖：Data Cleaning、EDA、Feature Engineering、Estimation and Inference、Hypothesis Testing、Correlation vs Causation、最终项目。

## 1. 总流程

~~~text
提出问题 -> 获取数据 -> 检查质量 -> 清理 -> EDA
-> 特征工程 -> 切 X/y -> 建模 -> 评估 -> 业务建议
~~~

X 是输入特征，y 是目标变量。真正决定模型质量的通常是数据清理、EDA 和特征工程。

## 2. Data Cleaning

脏数据包括：缺失值、重复值、不必要列、错误类型、文本不一致、异常值和数据泄漏。

~~~python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()
df.shape
df.info()
df.describe(include="all").T
df.dtypes
~~~

口诀：head 看样子，shape 看大小，info 看类型和非空，describe 看统计摘要。

缺失值：

~~~python
df.isna().sum().sort_values(ascending=False)
df.isna().mean().sort_values(ascending=False)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges_missing"] = (
    df["TotalCharges"].isna().astype(int)
)
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
df["Contract"] = df["Contract"].fillna("Unknown")
~~~

判断：缺失少且不重要可删除；数值列常用中位数；分类列可用众数或 Unknown；不能随便填 0。均值、中位数、标准差只能在训练集上 fit，再应用到测试集。

重复和类型：

~~~python
df.duplicated().sum()
df = df.drop_duplicates()
df.nunique().sort_values()
df = df.drop(columns=["customerID"])
df["gender"] = df["gender"].str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
~~~

先确认重复是否错误，再删除。ID 通常不是有意义的预测特征。

## 3. EDA

EDA 要回答：数据多大？变量什么类型？缺失和异常在哪里？目标如何分布？特征和目标有什么关系？特征之间是否共线？数据是否足够建模和检验？

~~~python
df.describe().T
df["target"].mean()
df["target"].median()
df["target"].quantile([0.25, 0.5, 0.75])

sample = df.sample(n=5, random_state=42)
df.groupby("species").size()
df.groupby("species")["sepal_length"].mean()
df["category"].value_counts()
df.corr(numeric_only=True)
~~~

中心：均值、中位数、众数。离散：方差、标准差、IQR。范围：最小值、最大值、分位数。

### 可视化

~~~python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# 散点图
plt.plot(
    df["sepal_length"],
    df["sepal_width"],
    linestyle="",
    marker="o",
    label="sepal"
)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Relationship")
plt.legend()

# 直方图
df["petal_length"].plot.hist(bins=25)
df.plot.hist(bins=25, alpha=0.5)
df.hist(figsize=(8, 8))

# 箱线图
df.boxplot(column="petal_length", by="species")
sns.boxplot(data=df, x="species", y="petal_length")

# Pairplot 和 Hexbin
sns.pairplot(df[features + ["target"]], hue="target")
sns.jointplot(
    data=df, x="x1", y="x2", kind="hex"
)

# FacetGrid
g = sns.FacetGrid(df, col="species")
g.map_dataframe(sns.histplot, x="sepal_length")
~~~

直方图看分布；箱线图看中位数、IQR 和异常值；散点图看两个连续变量；pairplot 看整体关系；hexbin 看二维密度。fig 是整张图，ax 是坐标区域。

~~~python
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["petal_length"], bins=25)
ax.set(
    xlabel="Petal length",
    ylabel="Frequency",
    title="Distribution"
)
~~~

### 异常值和残差

不要看到异常值就删除。先判断它是错误、真实极端值，还是研究本身的一部分。

~~~text
残差 = 实际值 - 预测值
标准化残差 = 残差 / 残差标准误差
~~~

标准化残差可以比较不同尺度下的偏离程度。处理方式：核对、修正、删除、截尾、变换、使用稳健模型或保留并说明。

## 4. Feature Engineering

四类工作：变量转换、编码、缩放、交互特征。

线性回归：

~~~text
y = beta_0 + beta_1*x1 + beta_2*x2 + error
~~~

beta_0 是截距，beta 是系数。

### 变量转换

~~~python
import numpy as np

df["income_log"] = np.log(df["income"])
df["income_log1p"] = np.log1p(df["income"])
~~~

右偏、长尾、回报递减时常用 log；有 0 时用 log1p，因为 log1p(x) 等于 log(1+x)。

~~~python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2, include_bias=False
)
X_poly = poly.fit_transform(X)
poly.get_feature_names_out(["x1", "x2"])
~~~

二次特征可能包括 x1、x2、x1²、x1*x2、x2²。加入多项式后，对参数仍是线性模型。

### 编码

~~~python
# 二进制
df["is_female"] = df["gender"].map({
    "Male": 0, "Female": 1
})

# 名义数据：one-hot
df = pd.get_dummies(
    df,
    columns=["Contract", "PaymentMethod"],
    dtype=int
)
~~~

名义数据没有顺序，不能把 Red、Blue、Green 随意编码为 1、2、3。

~~~python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[["low", "medium", "high"]]
)
df[["risk_code"]] = encoder.fit_transform(
    df[["risk"]]
)
~~~

序数编码有真实顺序，但会暗示距离相等，必须根据业务含义选择。

低频类别：

~~~python
counts = df["Neighborhood"].value_counts()
rare = counts[counts < 8].index
df["Neighborhood"] = df["Neighborhood"].replace(
    rare, "Other"
)
~~~

交互和分组内偏差：

~~~python
df["quality_year"] = (
    df["overall_quality"] * df["year_built"]
)
group_mean = df.groupby(
    "Neighborhood"
)["OverallQual"].transform("mean")
group_std = df.groupby(
    "Neighborhood"
)["OverallQual"].transform("std")
df["quality_group_z"] = (
    df["OverallQual"] - group_mean
) / group_std
~~~

transform 保持原行数，agg 通常返回缩小后的分组摘要。

### 缩放

~~~text
StandardScaler： (x - mean) / std，z-score
MinMaxScaler：   (x - min) / (max - min)，通常 0 到 1
RobustScaler：   (x - median) / IQR，抗异常值
~~~

~~~python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
~~~

口诀：训练集 fit_transform，测试集只 transform。

推荐顺序：统一类型 -> 缺失 -> 重复/无用列 -> 切训练测试 -> 训练集 fit 编码器和缩放器 -> transform 测试集 -> 检查输出。

## 5. 参数统计和常见分布

- 参数模型：依赖分布假设，只需有限参数，例如正态分布的均值和标准差。
- 非参数模型：较少依赖固定分布，更灵活，但通常需要更多数据。
- MLE：寻找最可能产生观察数据的参数。

~~~text
Uniform：范围内各值机会相同
Normal：均值附近最多，两边对称
Log-normal：取 log 后接近正态，常见于收入
Exponential：等待下一个事件的时间
Poisson：固定时间发生几次，均值 = 方差 = lambda
~~~

中心极限定理：很多随机样本的均值，在样本够大时趋近正态分布。

## 6. 贝叶斯和假设检验

~~~text
Posterior ∝ Likelihood × Prior
后验 ∝ 似然 × 先验
~~~

Prior 是看数据前的相信程度；Likelihood 是假设产生当前数据的可能性；Posterior 是看数据后的更新。

经典检验：

~~~text
提出 H0/H1
 -> 计算检验统计量
 -> 得到 H0 为真时的零分布
 -> 计算 p-value
 -> 与 alpha 比较
 -> 拒绝或不能拒绝 H0
~~~

- H0：没有效果、没有差异、等于具体值。
- H1：有差异、有影响、大于、小于或不等于。
- p < alpha：拒绝 H0。
- p >= alpha：不能拒绝 H0。

不要说证明 H0；要说证据不足以拒绝 H0。p-value 不是 H0 为真的概率，也不代表效果大小。

### 错误和术语

- Test statistic：用样本计算的决定性数字。
- Null distribution：H0 正确时统计量的分布。
- Rejection region：落入后拒绝 H0 的区域。
- I 型错误：H0 真的却拒绝它，假阳性。
- II 型错误：H0 是假的却没有拒绝它，假阴性。
- Power = 1 - II 型错误概率。

拒绝门槛越宽，power 可能更高但 I 型错误增加；门槛越严，I 型错误减少但漏检增加；增加样本量通常能减少权衡。

### F 统计量和 Bonferroni

F 检验常见 H0：所有回归系数为 0，加入特征不比只用目标均值更好。F 的 p-value 很小，说明整体回归有统计证据，不代表每个特征都重要，也不证明因果。

~~~python
prob_at_least_one = (
    1 - (1 - alpha) ** number_of_tests
)
alpha_adjusted = 0.05 / number_of_tests
~~~

检验越多，至少一次 I 型错误的概率越高。Bonferroni 降低 I 型错误，但牺牲 power。

### 二项和正态代码

~~~python
from scipy.stats import binom

prob_at_most_k = binom.cdf(k, n, p)
prob_at_least_k_plus_one = 1 - binom.cdf(k, n, p)
critical = binom.ppf(0.95, n, p)
~~~

cdf 是到 k 为止；1-cdf(k) 是比 k 更大。

~~~python
import numpy as np
from scipy import stats

x = np.linspace(1, 100, 200)
y = stats.norm.pdf(
    x, loc=50, scale=np.sqrt(10)
)
~~~

## 7. 相关性和因果性

相关性表示 X 能帮助预测 Y；因果性表示改变 X 会改变 Y。

可能关系：X 导致 Y；Y 导致 X；混杂变量 Z 同时导致 X 和 Y；随机巧合或时间趋势造成虚假相关。

例子：冰淇淋销量和溺水人数的混杂变量是温度；车祸和叫 John 的人数的混杂变量是人口规模；满意度低可能导致客服来电多，而不是客服来电导致满意度低。

判断因果要问：有没有机制？时间顺序对不对？是否控制混杂变量？是否有随机实验或准实验？能否在不同样本和时间中复现？

## 8. 项目和考试清单

项目必须包含：数据摘要、探索计划、清理过程、EDA 图表、特征工程、至少三个假设、至少一个显著性检验、主要发现、业务建议、限制和下一步。

~~~text
[ ] 数据规模、变量和目标清楚
[ ] 缺失值有理由，编码前后有证据
[ ] 图表有标题、轴标签和解释
[ ] 缩放只在训练集 fit
[ ] H0/H1、p-value、alpha 写对
[ ] 没有把相关性写成因果
[ ] 至少三个假设和一个显著性检验
[ ] 有行动建议、限制和下一步
~~~

## 9. 最小背诵代码

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats

df = pd.read_csv("data.csv")
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.describe().T)

df["x"] = pd.to_numeric(
    df["x"], errors="coerce"
)
df["x_missing"] = df["x"].isna().astype(int)
df["x"] = df["x"].fillna(df["x"].median())

sns.pairplot(df, hue="target")
sns.boxplot(data=df, x="category", y="x")

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

X = pd.get_dummies(
    X, columns=["category"], dtype=int
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alpha = 0.05
statistic, p_value = stats.ttest_ind(
    group_a, group_b, equal_var=False
)
decision = (
    "reject H0"
    if p_value < alpha
    else "fail to reject H0"
)
~~~

背诵顺序：检查 -> 清理 -> 可视化 -> 切 X/y -> 编码 -> 缩放 -> 检验 -> 解释。

## 一句话总结

~~~text
先确认数据可信，再用 EDA 发现结构；
用特征工程把变量变成模型能理解的形式；
用统计检验区分证据和偶然；
最后把结果翻译成可执行的业务行动。
~~~

