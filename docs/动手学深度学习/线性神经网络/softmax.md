# softmax

## softmax回归

回归也可以用于预测多少的问题，比如说房屋被售出的价格。事实上，我们也对分类问题感兴趣，比如说想问这个图像绘制的是驴、猫、狗还是鸡。回归估计一个连续值，而分类预测一个离散类别，例如MNIST数据集是手写数字识别（10类），ImageNet自然物体分类（1000类）。回归和分类有很多的相似性，但是又有区别。回归是单连续数值输出，跟真实值的区别作为损失；而分类有过个输出，输出i是预测为第i类的置信度。

那么分类中如何定义损失呢？我们使用均方损失。其中，假如说有n类，那么数据集里面的真实数据是对类别进行一位有效编码的结果，下面矩阵中$y_i=1,if\ i=y$, 而其余元素都是0。这种编码方式又称为独热编码（one-hot）:
$$
\mathbf{y}=[y_1, \dots, y_n]^T
$$
希望最后输出的值是一个个概率，然后希望这些概率都是非负，然后和为1。那么有没有什么操作满足这些需求呢？ softmax操作就可以。
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}
$$
那么有了输出值和真实值两个矩阵，如何设计损失函数呢？softmax函数给出了一个向量$\hat{\mathbf{y}}$，我们可以将其视为“对给定任意输入$\mathbf{x}$的每个类的条件概率”。例如，$\hat{y}_1$=$P(y=\text{猫} \mid \mathbf{x})$。假设整个数据集$\{\mathbf{X}, \mathbf{Y}\}$具有$n$个样本，其中索引$i$的样本由特征向量$\mathbf{x}^{(i)}$和独热标签向量$\mathbf{y}^{(i)}$组成。我们可以将估计值与实际值进行比较：
$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$
根据最大似然估计，我们最大化$P(\mathbf{Y} \mid \mathbf{X})$，相当于最小化负对数似然：
$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)})
$$
其中，对于任何标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，损失函数为：
$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
$$
那么假如说$\hat{\mathbf{y}}$是由$\mathbf{o}$矩阵经过softmax得来，那么如何求关于一个位置$o_j$的导数呢？
$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j
$$

换句话说，导数是我们softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。从这个意义上讲，这与我们在回归中看到的非常相似，其中梯度是观测值$y$和估计值$\hat{y}$之间的差异。

## 信息论审视角度

**信息论**（information theory）涉及编码、解码、发送以及尽可能简洁地处理信息或数据。信息论的核心思想是量化数据中的信息内容。在信息论中，该数值被称为分布$P$的**熵**（entropy）。可以通过以下方程得到：
$$
H[P] = \sum_j - P(j) \log P(j)
$$
信息论的基本定理之一指出，为了对从分布$p$中随机抽取的数据进行编码，我们至少需要$H[P]$“纳特（nat）”对其进行编码。“纳特”相当于**比特**（bit），但是对数底为$e$而不是2。因此，一个纳特是$\frac{1}{\log(2)} \approx 1.44$比特。

压缩与预测有什么关系呢？想象一下，我们有一个要压缩的数据流。如果我们很容易预测下一个数据，那么这个数据就很容易压缩。为什么呢？举一个极端的例子，假如数据流中的每个数据完全相同，这会是一个非常无聊的数据流。由于它们总是相同的，我们总是知道下一个数据是什么。所以，为了传递数据流的内容，我们不必传输任何信息。也就是说，“下一个数据是xx”这个事件毫无信息量。

但是，如果我们不能完全预测每一个事件，那么我们有时可能会感到"惊异"。克劳德·香农决定用信息量$\log \frac{1}{P(j)} = -\log P(j)$来量化这种惊异程度。***在观察一个事件$j$时，并赋予它（主观）概率$P(j)$。当我们赋予一个事件较低的概率时，我们的惊异会更大，该事件的信息量也就更大。***

如果把熵$H(P)$想象为“知道真实概率的人所经历的惊异程度”，那么什么是交叉熵？交叉熵**从**$P$**到**$Q$，记为$H(P, Q)$。我们可以把交叉熵想象为“主观概率为$Q$的观察者在看到根据概率$P$生成的数据时的预期惊异”。当$P=Q$时，交叉熵达到最低。在这种情况下，从$P$到$Q$的交叉熵是$H(P, P)= H(P)$。

简而言之，我们可以从两方面来考虑交叉熵分类目标：（i）最大化观测数据的似然；（ii）最小化传达标签所需的惊异。

## 图片分类数据集

Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。

以下函数用于在数字标签索引及其文本名称之间进行转换。

````python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
# 用svg高清显示图片
from d2l import torch as d2l
d2l.use_svg_display()

# 简单的预处理：所有的数据转化为张量; 注意方法是从torchvision中的transforms中来的
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

len(mnist_train), len(mnist_test)
# (60000, 10000)
mnist_train[0][0].shape
# torch.Size([1, 28, 28]) 代表通道为1（灰度图像），28*28代表长28像素宽28像素

def get_fashion_mnist_labels(labels): 
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
````

为了使我们在读取训练集和测试集时更容易，我们使用内置的数据迭代器，而不是从零开始创建。回顾一下，在每次迭代中，数据加载器每次都会[**读取一小批量数据，大小为`batch_size`**]。通过内置数据迭代器，我们可以随机打乱了所有样本，从而无偏见地读取小批量。

````python
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
# 4.3 sec
````

这里进程数是什么意思？数据要从硬盘转移到内存里面，这是一件不容易的事情，因此可能需要多进程来帮助数据更快地转移。实战中，建议单独检查读取一轮的数据的时间是多少，希望读取的时间至少要比训练的时间少，当然少很多是最好的。

现在就能整合所有的组件了：

````python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        # 上一步是为了把图片放大
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
# torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
````

## 从零开始实现的softmax

````python
import torch
from IPython import display
from d2l import torch as d2l
# 加载数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
# 初始化权重和偏置
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
def net(X):
    # torch.matmul是矩阵乘法；X.reshape是将bs*28*28 => bs * 784
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
````

上面我们进行了w b参数的初始化，定义了对于一个向量的softmax操作，定义了网络的流程，定义了交叉熵损失。给定预测概率分布`y_hat`，当我们必须输出硬预测（hard prediction）时，我们通常选择预测概率最高的类。当预测与标签分类`y`一致时，即是正确的。分类精度即正确预测数量与总预测数量之比。为了计算精度，我们执行以下操作。首先，如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数。我们使用`argmax`获得每行中最大元素的索引来获得预测类别。然后我们[**将预测类别与真实`y`元素进行比较**]。***由于等式运算符“`==`”对数据类型很敏感，因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致***。结果是一个包含0（错）和1（对）的布尔张量。最后，我们求和会得到正确预测的数量。

````python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
````

这里再定义Accumulator类，记录两个数字：预测正确的个数，和预测的次数。在下面的代码中，精度就是第一个数和第二个数字的比值。为什么这里使用了0.0？因为如果都是int，那么两个整数相除得到的结果就是int，况且accuracy返回的值也是规定为了float。

````python
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
````

训练一个epoch的代码如下：

````python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
````

在展示训练函数的实现之前，我们[**定义一个在动画中绘制数据的实用程序类**]`Animator`，这里就不放出来了。那么训练的完整过程函数如下：

````python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
````

作为一个从零开始的实现，我们使用[**小批量随机梯度下降来优化模型的损失函数**]，设置学习率为0.1。那么使用先前定义的网络、训练集和测试集的迭代器、损失函数、规定的训练轮数和参数更新器，我们可以开始训练了。

````python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
````

## softmax简洁实现

1. 这里的net相当于是两层操作，第一层是调整tensor的形状。`nn.Flatten()` 是 PyTorch 中用于将多维的输入张量展平成一维张量的层，形状为 `(batch_size, 28, 28)`，`nn.Flatten()` 会将每个样本展平为一维，变成 `(batch_size, 784)`（28×28 = 784）。这样处理之后，就可以将展平后的张量输入到线性层 `nn.Linear(784, 10)`，进行分类等操作。而第二层就是784维输入，输出10维的张量。
2. softmax函数$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中$\hat y_j$是预测的概率分布。$o_j$是未规范化的预测$\mathbf{o}$的第$j$个元素。如果$o_k$中的一些数值非常大，那么$\exp(o_k)$可能大于数据类型容许的最大数字，即**上溢**（overflow）。这将使分母或分子变为`inf`（无穷大），最后得到的是0、`inf`或`nan`（不是数字）的$\hat y_j$。解决这个问题的一个技巧是：在继续softmax计算之前，先从所有$o_k$中减去$\max(o_k)$。这看起来没啥特别，但实际上，这一步**不会改变 softmax 的输出**！因为指数函数只是相对大小的比较，减去一个常数后，比例不变。通过这一步，你可以避免 exponentiation 导致的数值过大，从而避免上溢问题。

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$

3. 在减法和规范化步骤之后，***可能有些$o_j - \max(o_k)$具有较大的负值***。由于精度受限，$\exp(o_j - \max(o_k))$将有接近零的值，即**下溢**（underflow）。这些值可能会四舍五入为零，使$\hat y_j$为零，并且使得$\log(\hat y_j)$的值为`-inf`。反向传播几步后，我们可能会发现自己面对一屏幕可怕的`nan`结果。为了避免上溢和下溢带来的问题，我们可以**结合 softmax 和交叉熵**，直接对未规范化的输出 `o_j` 进行处理，而不是先计算 softmax 再取对数。通过这个技巧，我们可以在计算交叉熵时跳过对 `exp` 函数的使用，避免潜在的数值稳定性问题。公式如下：

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

4. CrossEntropyLoss公式如下：

$$
L_i = -\log\left(\frac{\exp(o_{i, y_i})}{\sum_{j=1}^{C} \exp(o_{i, j})}\right)
= -o_{i, y_i} + \log\left(\sum_{j=1}^{C} \exp(o_{i, j})\right)
$$

5. `reduction = 'none'`，则返回每个样本的损失；`'mean'`返回的是整体损失平均；`'sum'`返回的是整体损失。换而言之，none返回的张量的形状是`torch.Tensor(batch_size, )`因为有batch_size个样本，每一个样本有一个损失值。这样的话，方便我们对这些Loss做一些torch API中没有设计的操作。

````python
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
# 参数相关的内容，都是torch.optim中的一些类来进行管理！
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
````



