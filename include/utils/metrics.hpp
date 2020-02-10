#ifndef MLKIT_METRICS_HPP
#define MLKIT_METRICS_HPP

namespace mk{
    namespace utils
    {
        inline af::array minkowski_distance(af::array & X, af::array & Y, int p=2)
        {
            int m = X.dims(0), n = Y.dims(0);
            af::array dist(m, n);
            for(int j = 0; j < n; ++j)
            {
                dist.col(j) = af::sum(af::pow(af::abs(batchFunc(X, Y.row(j), utils::bsub)), p), 1).T();
            }
            return dist;
        }
    }
}


#endif