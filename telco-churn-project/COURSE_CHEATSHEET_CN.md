# IBM EDA for ML：考前背诵清单

作者：yanyixuan

## 代码口诀

~~~text
读：read_csv
看：head / shape / info / describe
查：isna / duplicated / dtypes
清：dropna / fillna / drop_duplicates
画：hist / boxplot / scatter / pairplot
分：groupby / value_counts
变：log1p / PolynomialFeatures
编：get_dummies / OrdinalEncoder
缩：Standard / MinMax / Robust
切：X / y，再 train_test_split
检：statistic、p-value、alpha
说：不能拒绝 H0，不要说证明 H0
~~~

## 判断口诀

~~~text
Standard：减均值除标准差，z-score
MinMax：减最小除范围，通常 0 到 1
Robust：减中位数除 IQR，抗异常值

Male/Female -> 0/1
Red/Blue/Green -> one-hot
Low/Medium/High -> ordinal

H0：没有差异、没有效果、等于具体值
H1：有差异、有效果、大于、小于或不等于
p < alpha -> 拒绝 H0
p >= alpha -> 不能拒绝 H0
I 型：错误拒绝真的 H0
II 型：没有拒绝假的 H0
Power = 1 - II 型错误概率
~~~

## 分布和贝叶斯

~~~text
Uniform：范围内一样可能
Normal：均值附近最多，两边对称
Log-normal：取 log 后像正态
Exponential：等待下一个事件多久
Poisson：固定时间发生几次，均值 = 方差 = lambda

后验 ∝ 似然 × 先验
Posterior ∝ Likelihood × Prior
~~~

## 相关和因果

~~~text
相关：X 能帮助预测 Y
因果：改变 X 会改变 Y
先排查：反向因果、混杂变量、随机巧合、时间趋势
~~~

## 项目报告

~~~text
数据摘要 -> 探索计划 -> 清理前后对比 -> EDA
-> 特征工程 -> 三个假设 -> 显著性检验
-> 主要发现 -> 业务建议 -> 限制和下一步
~~~

