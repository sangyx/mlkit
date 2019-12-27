#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP
namespace mk{
    namespace linear_model{
        class LogisticRegression
        {
            public:
                af::array coef_, intercept_, n_iter_;
                LogisticRegression(double lr=0.1, std::string penalty="l2", double tol=0.0001, double C=1.0, bool fit_intercept=true, int max_iter=100, bool verbose=true);
                void fit(af::array & X, af::array & y);
                af::array predict(af::array & X);
                af::array predict_proba(af::array & X);
                float score(af::array & X, af::array & y);
            private:
                std::string penalty_;
                bool fit_intercept_, verbose_;
                double lr_, tol_, C_;
                int max_iter_;
        };

        inline LogisticRegression::LogisticRegression(double lr, std::string penalty, double tol, double C, bool fit_intercept, int max_iter, bool verbose): lr_(lr), penalty_(penalty), tol_(tol), C_(C), fit_intercept_(fit_intercept), max_iter_(max_iter), verbose_(verbose)
        {

        }

        inline void LogisticRegression::fit(af::array & X, af::array & y)
        {
            af::array X_copy = X.copy();
            int m = y.dims(0);

            if(this->fit_intercept_)
            {
                X_copy = af::join(1, af::constant(1, m, 1), X_copy);
            }

            af::array weight = af::randn(X_copy.dims(1), 1);
            int iter;
            float err;
            for(iter = 0; iter < this->max_iter_; ++iter){
                af::array h = af::sigmoid(af::matmul(X_copy, weight));
                err = af::max<float>(-af::sum(y * af::log(h) + (1 - y) * af::log(1 - h)));
                if(err < this->tol_)
                    break;

                if(this->verbose_ && (iter + 1) % 10 == 0)
                {
                    std::cout << "Iteration " << (iter + 1) << " Err: " << err << std::endl;
                }

                weight -= this->lr_ * af::matmulTN(X_copy, h - y) / m;
            }
            std::cout << "Training stopped after " << (iter + 1) << " iterations" << std::endl;
            this->coef_ = weight(af::seq(0, 1), af::span);
            this->intercept_ = weight(0);
        }

        inline af::array LogisticRegression::predict(af::array & X)
        {
            af::array prob = predict_proba(X);
            af::array y_hat = af::constant(0, X.dims(0), 1);
            af::replace(y_hat, prob > 0.5, 1);
            return y_hat;
        }

        inline af::array LogisticRegression::predict_proba(af::array & X)
        {
            if(this->fit_intercept_)
            {
                return af::sigmoid(af::matmul(X, this->coef_) + this->intercept_);
            }

            return af::sigmoid(af::matmul(X, this->coef_));
        }

        inline float LogisticRegression::score(af::array & X, af::array & y)
        {
            af::array h = predict_proba(X);
            return af::max<float>(-af::sum(y * af::log(h) + (1 - y) * af::log(1 - h)));
        }
    }
}
#endif