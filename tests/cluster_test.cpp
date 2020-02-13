#include <gtest/gtest.h>
#include <mlkit.hpp>

using namespace mk;
using namespace af;

TEST(ClusterTestCase, KMeansTest)
{
    array X = utils::load_dataset<float>("../tests/data/kmeans.txt");
    cluster::KMeans km(4);
    km.fit(X);
    EXPECT_NEAR(km.inertia_, 150, 1);
}