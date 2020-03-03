#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace af;
using namespace mk;

TEST(SVMTestCase, LinearSVCTest)
{
    array data = utils::load_dataset<float>("../tests/data/svm.txt");
    int feat = data.dims(1) - 1;
    array X = data.cols(0, feat-1);
    array y = data.col(feat);
    svm::LinearSVC svc;
    svc.fit(X, y);
    af_print(svc.predict(X));
    af_print(y);
}