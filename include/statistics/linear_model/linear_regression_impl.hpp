#ifndef LINEAR_REGRESSION_IMPL_HPP
#define LINEAR_REGRESSION_IMPL_HPP

#include "linear_regression.hpp"

namespace mk{
    namespace linear_model{
        inline LinearRegression::LinearRegression(const bool fit_intercept, const bool normalize): fit_intercept_(fit_intercept), normalize_(normalize){

        }

        inline void LinearRegression::fit(const af::array & X, const af::array & y){
            int m = y.dims(0);
            af::array X_copy = X.copy();
            if(this->fit_intercept_){
                X_copy = af::join(1, af::constant(1, m, 1), X_copy);
            }

            af::array weight = af::matmul(af::inverse(af::matmul(X_copy.T(), X_copy)), X_copy.T(), y);

            if(this->fit_intercept_){
                this->coef_ = weight(1, af::end);
                this->intercept_ = weight(0);
            }else{
                this->coef_ = weight;
            }
        }

        inline af::array LinearRegression::predict(const af::array & X) const{
            if(this->fit_intercept_)
                return af::matmul(X, this->coef_) + this->intercept_;

            return af::matmul(X, this->coef_);
        }

        inline double LinearRegression::score(const af::array& X, const af::array& y){
            af::array y_hat = this->predict(X);
            return af::matmul((y - y_hat).T(), y - y_hat).scalar<double>() / 2;
        }
    }
}
#endif