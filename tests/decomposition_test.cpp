#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace af;
using namespace mk;

TEST(DecompositionTestCase, PCATest)
{
    array data = utils::load_dataset<float>("../tests/data/iris.txt");
    int feat = data.dims(1) - 1;
    array X = data.cols(0, feat-1);
    array y = data.col(feat);
    decomposition::PCA pca(3);
    array Xd = pca.fit_transform(X);
    float* Xd_host = Xd.row(0).host<float>();
    EXPECT_NEAR(Xd_host[0], -2.68, 0.01);
    EXPECT_NEAR(Xd_host[1], -0.31, 0.01);
    EXPECT_NEAR(Xd_host[2], 0.02, 0.01);
}