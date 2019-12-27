#ifndef MLKIT_LINEAR_REGRESSION_HPP
#define MLKIT_LINEAR_REGRESSION_HPP
namespace mk{
    namespace linear_model{
        class LinearRegression{
            public:
                af::array coef_;
                af::array intercept_;

                LinearRegression(const bool fit_intercept=true, const bool normalize=false);

                void fit(const af::array& X, const af::array& y);
                af::array predict(const af::array& X) const;
                float score(const af::array& X, const af::array& y);
                // af::array get_params(bool deep);
                // af::array set_params();
            private:
                bool fit_intercept_, normalize_, copy_X_;
        };

        inline LinearRegression::LinearRegression(const bool fit_intercept, const bool normalize): fit_intercept_(fit_intercept), normalize_(normalize){

        }

        inline void LinearRegression::fit(const af::array & X, const af::array & y){
            int m = y.dims(0);
            af::array X_copy = X.copy();

            if(this->fit_intercept_){
                X_copy = af::join(1, af::constant(1, m, 1), X_copy);
            }

            if(!this->fit_intercept_ && this->normalize_){
                preprocessing::StandardScaler ss;
                X_copy = ss.fit_transform(X_copy);
            }

            af::array weight = af::matmul(af::inverse(af::matmul(X_copy.T(), X_copy)), X_copy.T(), y);

            if(this->fit_intercept_){
                this->coef_ = weight(af::seq(1, af::end), 0);
                this->intercept_ = weight(0, 0);
            }else{
                this->coef_ = weight;
            }
        }

        inline af::array LinearRegression::predict(const af::array & X) const{
            if(this->fit_intercept_)
                return batchFunc(af::matmul(X, this->coef_), this->intercept_, utils::badd);

            return af::matmul(X, this->coef_);
        }

        inline float LinearRegression::score(const af::array& X, const af::array& y){
            af::array y_hat = this->predict(X);
            return af::matmul((y - y_hat).T(), y - y_hat).scalar<float>() / 2;
        }
    }
}

#endif