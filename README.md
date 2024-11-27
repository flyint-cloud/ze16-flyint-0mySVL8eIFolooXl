
#### 简介


在上一篇文章《[机器学习：线性回归（上）](https://github.com)》中讨论了二维数据下的线性回归及求解方法，本节中我们将进一步的将其推广至高维情形。


#### 章节安排


1. 背景介绍
2. 最小二乘法
3. 梯度下降法
4. 程序实现


## 一、背景介绍


### 1\.1 超平面L的定义




---


定义在D维空间中的超平面L的方程为：


(1\.1\)L:wTx\+b\=0其中：wT\=\[w0,w1,…,wD]为不同维度的系数或权重，xT\=\[x0,x1,…,xD]为数据样本的特征向量。


在该定义中，超平面L是由是由法向量w和偏置项b决定的。具体来说，超平面L将D维空间划分为两个半空间，一个半空间满足wTx\+b\>0，另一个半空间满足wTx\+b\<0
，式(1\.1)称为矩阵表示法，也可以用标量表示法表示为：


(1\.2\)L:∑i\=1Dwixi\+b\=w1x1\+w2x2\+⋯\+wDxD\+b\=0在一些情况下，也会将偏置项b引入向量中，该方法分别对权重w和特征值x做增广：


xT\=\[1,x1,x2,…,xD]wT\=\[b,w1,w2,…,wD]在此基础上，超平面L的定义可以简化为：


(1\.3\)L:wTx\=0有时也简称


(1\.4\)L(x)\=0#### 示例


为方便读者理解，这里给出一个从二维的直线方程到超平面方程L的转换


y\=kx\+bkx−y\+b\=0\[bk−1]⋅\[1xy]\=0### 1\.2 高维线性回归




---


在高维线性回归任务中，采样数据的形式为S\={X,y}，其中X称为采样数据，为N×D的矩阵，y称为标签数据，更具体的有：


XT\=\[x0,x1,…,xN],xi\=\[xi1,xi2,…,xiD],xi∈RDyT\=\[y0,y1,…,yN]在高维数据的回归任务中，我们的目标是找到一个权重w，使得其能够对特征数据X给出预测y^


y^\=Xw其中：wT\=\[w1,…,wD]是大小为D∗1的向量。
同时，我们可以定义**均方根误差(MSE)**如下：


MSE\=‖y−Xw‖22其中‖⋅‖2为二范数，或欧几里得距离。
线性回归的目标为，最小化损失，下面我们将从最小二乘法和梯度下降法两个角度实现线性回归。


## 二、最小二乘法




---


最小二乘法（Least Squares Method）是一种广泛使用的线性回归问题的求解方法，其核心思想是，均方根误差MSE关于权重w的偏导为0时所求得的w为最优解，故对MSE化简如下：


MSE\=‖y−Xw‖22\=(y−Xw)T(y−Xw)\=yTy−wTXTy−yXw\+wTXTXw由于wTXTy和yXw是标量，其数值相等，故有：


MSE\=yTy−2wTXTy\+wTXTXw求MSE关于w的偏导得：


∂MSE∂w\=−2XTy\+2XTXw另偏导等于0得：


(2\.1\)XTy\=XTXw该方程称为**正规方程（Normal Equation）**，解该方程可得：


w\=(XTX)−1XTy### 2\.1 最小二乘法缺点


以下是最小二乘法的主要缺点：


**矩阵逆计算的复杂性**
最小二乘法的解析解需要计算矩阵XTX 的逆矩阵：


(2\.2\)w\=(XTX)−1XTy在高维情况下（即特征数量d较大），计算XTX 的逆矩阵的计算复杂度很高，甚至可能不可行。具体来说：


* 计算XTX的时间复杂度为O(nd2)，其中n是样本数量，d是特征数量。
* 计算矩阵逆的时间复杂度为O(d3)。


因此，当d很大时，计算逆矩阵的代价非常高。


**矩阵不可逆问题**


在高维情况下，特征数量d可能大于样本数量n，此时矩阵XTX可能是不可逆的（即奇异矩阵），这意味着无法直接计算其逆矩阵。此外，即使矩阵可逆，也可能因为浮点数精度问题导致计算结果不稳定。


**对异常值敏感**


最小二乘法对异常值非常敏感。因为最小二乘法最小化的是平方误差，所以异常值会对模型的拟合产生较大的影响。这可能导致模型的泛化能力下降。


**不适用于稀疏数据**


对于稀疏数据（即特征矩阵中有大量零元素），最小二乘法的计算效率较低。稀疏数据通常更适合使用稀疏矩阵的优化方法，如 Lasso 或 Ridge 回归。


**过拟合问题**


如果没有正则化，最小二乘法容易过拟合，尤其是在特征数量远大于样本数量的情况下。过拟合会导致模型在训练集上表现很好，但在测试集上表现很差。


**总结**


尽管最小二乘法在许多情况下是一个简单有效的线性回归求解方法，但它也存在一些明显的缺点，特别是在高维数据和复杂情况下。为了克服这些缺点，可以考虑使用其他优化方法，如梯度下降、岭回归（Ridge Regression）、Lasso 回归等，这些方法在计算效率、对异常值的鲁棒性和防止过拟合方面有更好的表现。


## 三、梯度下降法




---


梯度下降法是一种常用的优化算法。通过迭代更新模型的参数，使得均方误差逐步减小，最终达到最优解。


对于单个样本{xi,yi}，其损失函数定义为：


J(w)\=(y−xiw)2求其关于权重的偏导得：


∂∂wJ(w)\=∂∂w(y−xiw)2(3\.1\)\=2(y−xw)x故有参数修正公式如下：


(3\.2\)w:\=w−λ⋅∂J∂w## 四、程序实现


### 4\.1 生成测试数据




---


程序流程：


1. 定义特征维数`feature_num`及点个数`point_num`。
2. 定义权重向量`w`，特征数据`X`，标签数据`y`
3. 生成随机数，填充`w`和`X`
4. 定义误差向量`error`，并用随机数填充
5. 计算`y`



```


|  | #include |
| --- | --- |
|  | #include |
|  | #include |
|  |  |
|  | // Multiple linear regression data generation |
|  | namespace MLR { |
|  | void gen(Eigen::VectorXd& w, Eigen::MatrixXd& X, Eigen::VectorXd& y) { |
|  | if (w.rows() != X.cols()) { |
|  | throw std::invalid_argument("Dimension mismatch: The number of rows in w must equal the number of columns in X."); |
|  | } |
|  | if (X.rows() != y.rows()) { |
|  | throw std::invalid_argument("Dimension mismatch: The number of rows in X must equal the number of rows in y."); |
|  | } |
|  |  |
|  | w.setRandom(); |
|  | X.setRandom(); |
|  |  |
|  | Eigen::VectorXd error(y.rows()); |
|  | error.setRandom(); |
|  | error *= 0.02; |
|  |  |
|  | y = X * w + error; |
|  |  |
|  | return; |
|  | } |
|  | } |
|  |  |
|  |  |
|  | int main() { |
|  | const size_t point_num = 10; |
|  | const size_t feature_num = 7; |
|  |  |
|  | Eigen::VectorXd w(feature_num); |
|  | Eigen::MatrixXd X(point_num, feature_num); |
|  | Eigen::VectorXd y(point_num); |
|  |  |
|  | MLR::gen(w, X, y); |
|  |  |
|  | std::cout << "y =\n" << y << "\n"; |
|  |  |
|  | return 0; |
|  | } |


```

### 4\.2 最小二乘法实现：




---


程序流程：


1. 构建向量`wp`用以存储计算结果
2. 采用公式(2\.2)计算权重`wp`
3. 输出`w-wp`以观察计算误差



> Eigen库中求逆、求转置都需要以矩阵为主体，例如: `M.inverse()`和`M.transpose()`。



> 取名`wp`是因为Weight prediction的首字母。



```


|  | void LSM(Eigen::VectorXd& w, Eigen::MatrixXd& X, Eigen::VectorXd& y) { |
| --- | --- |
|  | if (w.rows() != X.cols()) { |
|  | throw std::invalid_argument("Dimension mismatch: The number of rows in w must equal the number of columns in X."); |
|  | } |
|  | if (X.rows() != y.rows()) { |
|  | throw std::invalid_argument("Dimension mismatch: The number of rows in X must equal the number of rows in y."); |
|  | } |
|  |  |
|  | w = (X.transpose() * X).inverse() * X.transpose() * y; |
|  | } |
|  |  |
|  | int main() { |
|  | // ... |
|  |  |
|  | Eigen::VectorXd wp(feature_num); |
|  |  |
|  | LSM(wp, X, y); |
|  |  |
|  | std::cout << "w_error =\n" << w-wp << "\n"; |
|  |  |
|  | return 0; |
|  | } |


```

下图为程序输出结果，由该图可以看出，最小二乘法的估计较为准确。
![description](https://img2024.cnblogs.com/blog/3320410/202411/3320410-20241126141716970-1037171205.png)


### 4\.3 梯度下降法实现




---


程序流程：


1. 构建向量`wp`，并初始化为随机权重。
2. 每一个数据样本`x`，依据公式(3\.2)更新一次权重。（`GD_step`函数功能）
3. 重复步骤2，100次。
4. 输出`w-wp`以观察计算误差


**注意事项：**



> 在该算法中，我们将样本的个数改为100个，即：`feature_num = 100`



> 学习率过高会导致发散，详细参考上一篇文章：《[机器学习：线性回归（上）](https://github.com):[PodHub豆荚加速器官方网站](https://rikeduke.com)》



> 下式子作用是将矩阵`X`的第`idx`行读取为列向量
> `Eigen::VectorXd x = X.row(idx);`
> 这与我们的使用直觉不符，实际上应为行向量。为避免出错，在后续计算中应使用`x.transpose()`而非直接使用`x`。
> 有一种方法可以规避该问题，即使用点积（内积）进行计算。在代码中给出了相关的示例（注释部分）



```


|  | void GD_step(Eigen::VectorXd& w, Eigen::MatrixXd& X, Eigen::VectorXd& y, const double& lambda) { |
| --- | --- |
|  | if (w.rows() != X.cols()) { |
|  | throw std::invalid_argument("Dimension mismatch: The number of rows in w must equal the number of columns in X."); |
|  | } |
|  | if (X.rows() != y.rows()) { |
|  | throw std::invalid_argument("Dimension mismatch: The number of rows in X must equal the number of rows in y."); |
|  | } |
|  |  |
|  | for (size_t idx = 0; idx < X.rows(); ++idx) { |
|  | Eigen::VectorXd x = X.row(idx); |
|  |  |
|  | // 使用点积 |
|  | // Eigen::VectorXd gradient = 2 * (y(idx) - x.dot(w)) * x; |
|  |  |
|  | // 因为 y-x*w是标量，且输出结果为VectorXd，因此最后的transpose是可去的。 |
|  | // Eigen::VectorXd gradient = 2 * (y(idx) - x.transpose() * w) * x.transpose(); |
|  |  |
|  | Eigen::VectorXd gradient = 2 * (y(idx) - x.transpose() * w) * x; |
|  |  |
|  | w += lambda * gradient; |
|  | } |
|  | } |
|  |  |
|  | int main() { |
|  | const size_t point_num = 100; |
|  |  |
|  | // ... |
|  |  |
|  | Eigen::VectorXd wp(feature_num); |
|  | wp.setRandom(); // 生成初始值 |
|  |  |
|  | double lambda = 2e-3; |
|  |  |
|  | for (int _ = 0; _ < 100; ++_) { |
|  | GD_step(wp, X, y, lambda); |
|  | } |
|  |  |
|  | std::cout << "w_error =\n" << w - wp << "\n"; |
|  |  |
|  | return 0; |
|  | } |


```

下图为程序输出结果，由该图可以看出，梯度下降法的估计较为准确。
![description](https://img2024.cnblogs.com/blog/3320410/202411/3320410-20241126144422950-1763099904.png)


