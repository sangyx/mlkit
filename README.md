# MLKIT
> A HEADER-ONLY LIBRARY PROVIDES SKLEARN-LIKE API WITH GPU SUPPORT.

## 依赖库
* [ArrayFire(矩阵计算)](http://arrayfire.org/)
* [gtest(单元测试)](https://github.com/google/googletest)

## 使用示例
```cpp
#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

int main(int argc, char **argv)
{
    array data = utils::load_dataset<float>("linear_regression.txt");
    array X = data.cols(0, 1);
    array y = data.col(2);
    linear_model::LinearRegression lr = linear_model::LinearRegression(false);
    lr.fit(X, y);
}
```

## ToDo
* 统计学习：
    - [ ] cluster.KMeans
    - [ ] decomposition.PCA
    - [ ] mixture.GaussianMixture
    - [ ] svm.LinearSVC, svm.LinearSVR
    - [ ] tree.DecisionTreeClassifier, tree.DecisionTreeRegressor
* 集成学习：
    - [ ] ensemble.AdaBoostClassifier
    - [ ] ensemble.AdaBoostRegressor

## 手记系列

### 数学基础
* [数学分析](https://www.sangyx.cn/281)
* [概率论与数理统计](https://www.sangyx.cn/1155)
* [线性代数](https://www.sangyx.cn/1161)

### 统计学习
* [线性回归](https://www.sangyx.cn/304)
* [KNN](https://www.sangyx.cn/1193)
* [决策树](https://www.sangyx.cn/1195)

### 优化
* [梯度下降](https://www.sangyx.cn/261)

## 参考
* 李航. 统计学习方法[M]. 2012.
* Peter Harrington. 机器学习实战[M]. 2013.
* 华校专. AI算法工程师手册[EB/OL]. <http://www.huaxiaozhuan.com/>.
* Christopher M Bishop. Pattern Recognition and Machine Learning[M]. 2006.
