#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace af;
using namespace mk;

TEST(MixtrueTestCase, GaussianMixtureTest)
{
    array data = utils::load_dataset<float>("../tests/data/iris.txt");
    int feat = data.dims(1) - 1;
    array X = data.cols(0, feat-1);
    mixture::GaussianMixture gm(3);
    gm.fit_predict(X);
    std::cout << gm.n_iter_ << std::endl;
    af_print(gm.weights_);
    af_print(gm.means_);
    af_print(gm.covariances_[0]);
}