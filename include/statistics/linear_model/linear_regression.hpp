#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP
namespace mk{
    namespace linear_model{
        class LinearRegression{
            public:
                af::array coef_;
                af::array intercept_;

                LinearRegression(const bool fit_intercept=true, const bool normalize=false);

                void fit(const af::array& X, const af::array& y);
                af::array predict(const af::array& X) const;
                double score(const af::array& X, const af::array& y);
                // af::array get_params(bool deep);
                // af::array set_params();
            private:
                bool fit_intercept_, normalize_, copy_X_;
        };
    }
}

#include "linear_regression_impl.hpp"

#endif