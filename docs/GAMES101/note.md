# GAMES101 Note

## Overview of Computer Graphics

Computer Graphic，简称CG，它的定义是：The use of computers to synthesize and manipulate visual information。“合成与操纵”。

现在很多游戏都利用CG技术，那么如何评价画面的好坏呢？一种简单的断言：看画面够不够亮。因为这涉及到CG的全局光照概念，因此将光照是否充足作为评价标准是可以的。

以及，不同的游戏可能有不同的画质，例如有的是卡通画质，有的是更贴切显示的画质，那么如何实现不同的画质呢？这些也都需要计算机图形学来解决。除开游戏，电影里面也广泛应用CG，如特效（special effect)，人物面部捕捉（阿凡达）等。除了电影游戏，很多其他领域也应用CG，例如Computer-Aided Design（如数字孪生 photo->CG），动画（animation），可视化（Visualization for Science, engineering, medicine, and journalism, etc)，虚拟现实（Virtual Reality）等。

图形学具体包含了哪些内容呢？

- Math of (perspective) projections, curves, surfaces
- Physics of lighting and shading
- Representing / operating shapes in 3D 
- Animation / simulation

课程主要包括四个方面内容：

- Rasterization （光栅化）
- Curves and Meshes （曲线和曲面）
- Ray Tracing （光线追踪）
- Animation / Simulation （动画与模拟）

什么是光栅化？把空间中的三维形体投射到屏幕上，就是光栅化：Project geometry primitives (3D triangles / polygons) onto the screen。这是**实时**计算机图形学的应用。CG中定义到：一秒钟能生成30张画面（30 frames per second）就认为是实时，否则就是**离线**（off-line）。

而在曲线和曲面中，将了解如何在计算机视觉中表示几何：How to present geometry in Computer Graphics。例如空间中如何表示较为光滑的曲面？如何用简单的面片去细粒度地拟合较为光滑的曲面？当物体改变时，曲面应该如何改变？物体改变时，如何保持原有的拓扑结构？

光线追踪被电影广泛使用，用来生成更高质量的画面。CG中也有一种trade-off: 生成速度快 or 生成质量高？那么光线追踪就是选择了质量高，但是生成时间很长。那么有没有两全其美的方法呢？最近有“实时光线追踪”技术。

动画与模拟努力将画面中物体的移动、变化等尽可能地贴切现实生活。例如rubber ball的弹跳，悬挂毛巾的下垂等等。

GAMES101不提到的：

- Using OpenGL / DirectX / Vulkan
- The syntax of Shaders
- We learn CG,  not Graphics APIs
- 3D modeling using Maya / 3DS MAX / Blender, or Unity / Unreal Engine
- CV

那么如何理解CV和CG的区别？

![image](img/1.png)

当然，现在两者的边界原来越模糊。自从NeRF(Neural Radiance Field，神经辐射场)的诞生，Model and Image的交融研究越来越多。

课程使用语言： C++

## Review of Linear Algebra

CG其实交叉了很多的学科：数学中涉及到线性代数，微积分和概率统计；物理学中涉及到光学和运动学；以及其它一些领域，如信号处理，数值分析，美学等。加下来将会光速回顾一些线性代数中的基础：

向量vector是基本（$\vec{a}$），它的范数（magnitude）是$\lvert\lvert\vec{a}\rvert\rvert$，而单位向量的范数自然是1。如果希望对于向量$\vec{a}$，得到对应的单位向量，直接$\hat{a} = \vec{a} / \lvert\lvert\vec{a}\rvert\rvert$即可。一般单位向量用来表示方向。向量相加满足三角形法则。

除了加减，还有点乘（dot / scalar product），得到的就是内积。点乘可以用来**快速计算**两个向量的夹角$cos\theta$，尤其是当两个向量都是方向向量的时候。同时，点乘还能帮助找到一个向量投射到另一个向量上的结果。最后，点乘结果能帮助判断向量之间夹角与直角的关系。

点乘的基本性质：交换律，结合律，等等。在笛卡尔坐标系下，点乘的计算会非常方便：
$$
\vec{a}\cdot\vec{b}
=
\begin{pmatrix}
x_a \\
y_a \\
z_a
\end{pmatrix}
\cdot
\begin{pmatrix}
x_b \\
y_b \\
z_b
\end{pmatrix}
=
x_ax_b + y_ay_b + z_az_b
\tag{3D}
$$
点乘之外，还有叉乘。两个向量叉乘（cross product）得到的结果是另一个向量，这个向量与两个向量都垂直（3D中），方向由右手定则决定。叉乘没有交换律。叉乘在建立三维空间直角坐标系中非常有用。有时候还能帮助判断两个向量谁在谁的左边/右边。
$$
\lvert\lvert\vec{a}×\vec{b}\rvert\rvert
=
\lvert\lvert\vec{a}\rvert\rvert
\lvert\lvert\vec{b}\rvert\rvert
sin\phi
$$
The Cartesian Formula of Cross Pruduct in 3D Euclidean Space:
$$
\vec{a}×\vec{b}\ 
=
\begin{pmatrix}
y_az_b - y_bz_a \\
z_ax_b - x_az_b \\
x_ay_b - y_ax_b
\end{pmatrix}
$$
那么有另一种表现形式（A为a向量的***dual matrix***）
$$
\vec{a}×\vec{b}\ = A*b = 
\begin{pmatrix}
0 & -z_a & y_a \\ 
z_a & 0 & -x_a \\ 
-y_a & x_a & 0
\end{pmatrix}
\begin{pmatrix}
x_b \\
y_b \\ 
z_b
\end{pmatrix}
$$
叉乘在CG的Rasterization光栅化中一个非常重要的应用，就是判断一个点在不在一个polygon里面，如判断是不是在三角形中。在下面这个例子中，只需要判断`AP * AB  BP * BC  CP * CA`，这三个叉乘的结果，就能知道P点在不在三角形里面，因为叉乘能帮忙判断P在AB BC CA三个向量的左边还是右边。值得注意的是，这种判断方式和三角形三条边顶点的绕向无关(AB BC CA or BA AC CB)，因为最终判断的是P点在不在三条向量的**同侧**（当然，三条向量首尾相接是需要保证的）。当然，如果点落在了边上，那么就属于corner case, 可以自己决定是否在三角形里面。

<img src="img/2.png" alt="image" style="zoom: 33%;" />

矩阵之间的乘法略。矩阵的转置有以下性质：$(AB)^T=B^TA^T$。那么点乘就可以表示为：$\vec{a}\cdot\vec{b} = \vec{a}^T\vec{b}$

## Eigen库

Eigen是一个高层次的C ++库，有效支持线性代数，矩阵和矢量运算，数值分析及其相关的算法。

关于Eigen库的安装，之前一直尝试用cMake方式把库编进MinGW里面，但是总是失败。于是最后阅读了[官方网站]([Eigen: Getting started](https://eigen.tuxfamily.org/dox/GettingStarted.html))，发现了如下的内容：
In order to use [Eigen](https://eigen.tuxfamily.org/dox/namespaceEigen.html), you just need to download and extract [Eigen](https://eigen.tuxfamily.org/dox/namespaceEigen.html)'s source code (see [the wiki](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) for download instructions). In fact, the header files in the `Eigen` subdirectory are the only files required to compile programs using [Eigen](https://eigen.tuxfamily.org/dox/namespaceEigen.html). The header files are the same for all platforms. It is not necessary to use CMake or install anything.

There is no library to link to. The only thing that you need to keep in mind when compiling the above program is that the compiler must be able to find the [Eigen](https://eigen.tuxfamily.org/dox/namespaceEigen.html) header files. The directory in which you placed [Eigen](https://eigen.tuxfamily.org/dox/namespaceEigen.html)'s source code must be in the include path. With GCC you use the `-I` option to achieve this:

`g++ -I /path/to/eigen/ my_program.cpp -o my_program `

于是命令行运行cpp文件的时候，应该终端运行上面指令，然后终端命令行运行exe文件：`./my_program.exe`

## Transform

















