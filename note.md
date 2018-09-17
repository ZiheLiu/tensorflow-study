## 记录

### Tensorflow

#### 常量、变量、占位符

```python
# placeholder要在run的时候feed值
tf.placeholder(tf.float32)

# 用初始化值初始化tf变量
tf.Variable(0.0)
# 用shape初始化tf变量
tf.Variable([1, 3])

# 常量
tf.constant(10)

# 根据高斯分布生成特定形状向量的随机值
tf.random_normal([10, 2])
```

####函数操作

```python
# 求平均值
tf.reduce_mean()

# 求平方
tf.square()

# 矩阵乘法
# [a, b] * [b, c] = [a, c]
tf.matmual(v1, v2)

# 加法
# v1的形状为[a, b], v2的形状为[1, b]
# v1[i][j] += v2[1][j]
# [[1， 2]， [3， 4]， [5， 6]] + [10， 11] = [[11， 13]， [13， 15]， [15， 17]]
v1 + v2

# 两个维度相同的向量，对应位置值相等为true，否则为false
# v1 = [[1, 2], [3, 4], [5, 6]]
# v2 = [[1, 2], [3, 40], [50, 6]]
# tf.equal(v1, v2) = [[True, True], [True, False], [False, True]]
tf.equal(v1, v2)

# 把向量v中的值转为第二个参数的类型
tf.cast(v, tf.float32)
```



### Numpy

```python
# 在[0, 1)范围内的随机数
np.random.rand()

# 从-0.5到0.5均匀的200个值
np.linspace(-0.5, 0.5, 200)

# 为向量增加一维度
data_x = np.array([1, 2, 3])
new_data_x = data_x[:, np.newaxis]
"""
得到
[[1],
 [2],
 [3]]
"""

# 根据高斯分布(正态分布)生成特定形状向量的随机点
# loc: 概率分布的均值
# scale 概率分布的标准差
# size shape
np.random.normal(loc=0.0, scale=1.0, size=None)
```



### Plt

```python
# 创建一个新的图表
plt.figure()

# 画散点
plt.scatter(x, y)

# 画线
# 'r-'红色实线
# 线的宽度为5
plt.plot(x, y, 'r-', lw=5)
```

