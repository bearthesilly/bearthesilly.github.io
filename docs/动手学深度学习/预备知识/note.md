# 预备知识

## 数据操作

N维数组是机器学习和神经网络的主要数据结构：0D是标量；1D是向量，比如说一个特征向量；2D是一个矩阵，比如说样本-特征矩阵；3D可以代表RGB图片，宽×高×通道；4D可以是一个RGB照片批量。诸如此类。

创建数组的时候，需要三个参数：形状，例如3×4矩阵；元素数据类型，例如说32位浮点数；每个元素的值，例如说全是0或者随机分布。那么访问元素用下标进行访问：正规来说，一个维度上，有三个参数，`a:b:c`，a代表起始位置，b代表终止位置但是不包含（即访问区间为$[a,b)$），c代表访问时的步长。如果只传入一个参数，那么默认填入a，两个就是ab，三个就是abc；三个参数没有必要全填满，步长默认是1，而a b没填的话代表前面or后面的全取。比如说：`[1:3, 1:]`在二维矩阵中就是行数为1 2，列数为1及以后的全部元素；`[::3, ::]`代表的就是行数上从0开始步长为3，列数上都取中的元素。

同时关于张量的一些基本操作如下：

````python
import torch
x = torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
x.shape
# torch.Size([12])
x.numel() # 张量中元素的总数
X = x.reshape(3, 4) # 改变形状但是不改变元素数量和值
X = x.reshape(-1, 4) # -1用来自动计算维度
x1 = torch.zeros((2, 3, 4)) # 传入的参数代表了形状维度，这个方法生成的值都是0
x2 = torch.ones((2, 3, 4)) # 同上，生成的全是1；注意传入的是元组
x3 = torch.randn(3, 4) # 随机生成浮点数，而且是在均值为0，标准差为1的高斯分布中随机采样
# 当然上面这个方法传入元组也是正确的
x4 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 利用嵌套列表来为所需张量的每一个元素赋值

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
'''
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
'''
torch.exp(x)
# tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])

X = torch.arange(12, dtype=torch.float32).reshape((3,4)) # 规定数据类型
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
# 注意dim参数代表拼接的维度
'''
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
'''
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=0)
'''
tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]])
'''
x = X == Y # 每一个位置，如果X和Y相等，那么构建的新张量对应位置是1
X.sum # 所有元素求和
````

其中，有一种特殊的广播机制，来自于Numpy的习惯：由于`a`和`b`分别是$3\times1$和$1\times2$矩阵，如果让它们相加，它们的形状不匹配。我们将两个矩阵*广播*为一个更大的$3\times2$矩阵，如下所示：矩阵`a`将复制列，
矩阵`b`将复制行，然后再按元素相加。

````python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a+b
'''
tensor([[0, 1],
        [1, 2],
        [2, 3]])
'''
````

同时如果希望反过来切片，需要利用`torch.flip(tensor, dims=[])`方法（dims里面传入的参数可以有多个）

````python
X = torch.arrange(12).reshape(3, 4)
torch.flip(X, dims=[1])
'''
tensor([[ 3.,  2.,  1.,  0.],
         [ 7.,  6.,  5.,  4.],
         [11., 10.,  9.,  8.]])
'''
torch.flip(X, dims=[0, 1])
'''
 tensor([[11., 10.,  9.,  8.],
         [ 7.,  6.,  5.,  4.],
         [ 3.,  2.,  1.,  0.]]))
'''
````

同时：**运行一些操作可能会导致为新结果分配内存**。例如，如果我们用`Y = X + Y`，我们将取消引用`Y`指向的张量，而是指向新分配的内存处的张量。

````python
before = id(Y)
Y = Y + X
id(Y) == before # False
````

这可能是不可取的，原因有两个：

首先，我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新；

其次，如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数。

但同时幸运的是，原地执行操作还是非常简单的。我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如`Y[:] = <expression>`。

````python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
# 两次打印的结果相同
````

**如果在后续计算中没有重复使用`X`，我们也可以使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销。**

````python
before = id(X)
X += Y
id(X) == before # True
````

将深度学习框架定义的张量**转换为NumPy张量（`ndarray`）**很容易，反之也同样容易。torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。

````python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
# (numpy.ndarray, torch.Tensor)
a = torch.tensor([3.5])
type(a[0:]), a.item(), type(a.item()), type(float(a)), int(a)
# (torch.Tensor, 3.5, float, float, 3)
````

## 数据预处理

为了能用深度学习来解决现实世界的问题，我们经常从预处理原始数据开始，而不是从那些准备好的张量格式数据开始。在Python中常用的数据分析工具中，我们通常使用`pandas`软件包。像庞大的Python生态系统中的许多其他扩展包一样，`pandas`可以与张量兼容。

````python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    
import pandas as pd
data = pd.read_csv(data_file)
print(data)
'''
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
'''
````

注意，“NaN”项代表缺失值。**为了处理缺失的数据，典型的方法包括*插值法*和*删除法*，**其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。在***这里，我们将考虑插值法***。

通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`，其中前者为`data`的前两列，而后者为`data`的最后一列。对于`inputs`中缺少的数值，我们用同一列的***均值***替换“NaN”项。

````python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
'''
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
'''
````

**对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。**由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，`pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

````python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
'''
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
'''
````

**现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。**当数据采用张量格式后，可以通过在 :numref:`sec_ndarray`中引入的那些张量函数来进一步操作。

````python
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
'''
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
'''
````

注意：这里的get_dummies方法只对一列数据中要么是字符串要么是NaN这种情况起作用。而且字符串一共几类，那么就会自动分出多少列。例如：假如说NumRooms里面修改成：NaN yes no三种，那么结果就会如下：

````python
'''
   NumRooms_no  NumRooms_yes  NumRooms_nan  Alley_Pave  Alley_nan
0            0             0             1           1          0
1            0             1             0           0          1
2            1             0             0           0          1
3            0             0             1           0          1
'''
````





