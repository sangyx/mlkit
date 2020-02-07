# MLKIT

[![Build Status](https://travis-ci.com/sangyx/mlkit.svg?branch=master)](https://travis-ci.com/sangyx/mlkit) [![codecov](https://codecov.io/gh/sangyx/mlkit/branch/master/graph/badge.svg)](https://codecov.io/gh/sangyx/mlkit)

> A HEADER-ONLY LIBRARY PROVIDES SKLEARN-LIKE API WITH GPU SUPPORT.

## 依赖库
* [ArrayFire(矩阵计算)](http://arrayfire.org/)
* [gtest(单元测试)](https://github.com/google/googletest)

## 使用示例
```cpp
#include "mlkit.hpp"

using namespace std;
using namespace mk;

int main(int argc, char **argv)
{
    int device = argc > 1 ? atoi(argv[1]) : -1; // default -1
    try {
        if(device >= 0)
            af::setBackend(AF_BACKEND_CUDA); // use gpu
        else
            af::setBackend(AF_BACKEND_CPU); // use cpu

        af::info();
        af::array X = af::randn(100, 3);
        af::array y = 1 * X(af::span, 0) + 2 * X(af::span, 1) + 3 * X(af::span, 2) + 4 + af::randu(100, 1) * 0.5;
        linear_model::LinearRegression lr = linear_model::LinearRegression(true);
        lr.fit(X, y);
        cout << endl \
             << "[linear regression]" << endl \
             << "-----------------------------------------------" << endl \
             << "expect coef: [1, 2, 3], expect intercept: 4" << endl \
             << "-----------------------------------------------" << endl \
             << "fit result: " << endl;
        af_print(lr.coef_);
        af_print(lr.intercept_)
        cout << "-----------------------------------------------" << endl \
             << "fit error: " << lr.score(X, y) << endl;
    } catch (af::exception &ae) {
        cerr << ae.what() << endl;
    }
    return 0;
}
```

This output would be:
```bash
# compiler command
g++ -std=c++11 -g example.cpp -o test -I/opt/arrayfire/include -I/usr/local/mlkit/include -laf -L/opt/arrayfire/lib

# output
ArrayFire v3.7.0 (CPU, 64-bit Linux, build c30d5455)
[0] Intel: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz, 95293 MB, Max threads(20) GNU Compiler Collection(GCC/G++) 7.4.0

[linear regression]
-----------------------------------------------
expect coef: [1, 2, 3], expect intercept: 4
-----------------------------------------------
fit result:
lr.coef_
[3 1 1 1]
   Offset: 1
   Strides: [1 4 4 4]
    0.9999
    1.9851
    2.9896

lr.intercept_
[1 1 1 1]
   Offset: 0
   Strides: [1 4 4 4]
    4.2475
-----------------------------------------------
fit error: 0.8954
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
* [逻辑回归](https://www.sangyx.cn/331)
* [KNN](https://www.sangyx.cn/1193)
* [决策树](https://www.sangyx.cn/1195)

### 优化
* [梯度下降](https://www.sangyx.cn/261)

## 参考
* 李航. 统计学习方法[M]. 2012.
* Peter Harrington. 机器学习实战[M]. 2013.
* 华校专. AI算法工程师手册[EB/OL]. <http://www.huaxiaozhuan.com/>.
* Christopher M Bishop. Pattern Recognition and Machine Learning[M]. 2006.
