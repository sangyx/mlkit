#ifndef MLKIT_GAUSSIAN_MIXTURE
#define MLKIT_GAUSSIAN_MIXTURE
namespace mk
{
    namespace mixture
    {
        class GaussianMixture
        {
        private:
            int n_components_, max_iter_;
            float tol_;
        public:
            af::array weights_, means_;
            std::vector<af::array> covariances_;
            bool converged_;
            int n_iter_;
            GaussianMixture(int n_components=1, float tol=0.001, int max_iter=100);
            void fit(af::array & X);
            af::array predict(af::array & X);
            af::array predict_proba(af::array & X);
            af::array fit_predict(af::array & X);
        };

        inline GaussianMixture::GaussianMixture(int n_components, float tol, int max_iter): n_components_(n_components), tol_(tol), max_iter_(max_iter)
        {
        }

        inline void GaussianMixture::fit(af::array & X)
        {
            int n_samples = X.dims(0), n_features = X.dims(1);
            this->converged_ = false;
            this->weights_ = af::constant(1.0 / this->n_components_, this->n_components_);
            this->means_ = af::randu(this->n_components_, n_features);

            for(int k = 0; k < this->n_components_; ++k)
            {
                af::array cov = af::upper(af::randu(n_features, n_features) * 100);
                for(int i = 0; i < n_features; ++i)
                {
                    for(int j = 0; j < i; ++j)
                    {
                        cov(i, j) = cov(j, i);
                    }
                }
                this->covariances_.push_back(cov);
            }

            // af_print(this->covariances_[0]);

            for(this->n_iter_ = 0; this->n_iter_ < this->max_iter_; ++this->n_iter_)
            {
                af::array gamma(n_samples, this->n_components_);
                for(int j = 0; j < n_samples; ++j)
                {
                    for(int k = 0; k < this->n_components_; ++k)
                    {
                        af::array z = af::batchFunc(X.row(j), this->means_.row(k), utils::bsub);

                        gamma(j, k) = this->weights_.row(k) * af::exp(-af::matmul(z, af::inverse(this->covariances_[k]), z.T()) / 2.0) / (sqrt(pow(2 * M_PI, n_features) * abs(af::det<float>(this->covariances_[k]))));
                    }
                }

                gamma = af::batchFunc(gamma, af::sum(gamma, 1), utils::bdiv);

                af::array means = af::constant(0, this->n_components_, n_features);
                af::array weights = af::constant(0, this->n_components_);
                std::vector<af::array> covariances;
                for(int k = 0; k < this->n_components_; ++k)
                {
                    af::array cov = af::constant(0, n_features, n_features);
                    for(int j = 0; j < n_samples; ++j)
                    {
                        af::array z = af::batchFunc(X.row(j), this->means_.row(k), utils::bsub);
                        means.row(k) += af::batchFunc(X.row(j), gamma(j, k), utils::bmul);
                        cov += af::batchFunc(af::matmul(z.T(), z), gamma(j, k), utils::bmul);
                        weights.row(k) += gamma(j, k);
                    }

                    means.row(k) /= af::sum<float>(gamma.col(k));
                    cov /= af::sum<float>(gamma.col(k));
                    weights.row(k) /= n_samples;
                    covariances.push_back(cov);
                }

                if(af::max<float>(af::abs(means - this->means_)) < this->tol_)
                    this->converged_ = true;
                this->means_ = means;
                this->weights_ = weights;
                this->covariances_ = covariances;
                if(this->converged_)
                    break;
            }
        }

        inline af::array GaussianMixture::predict(af::array & X)
        {
            af::array gamma = this->predict_proba(X);
            af::array y_hat, values;
            af::topk(values, y_hat, gamma.T(), 1);
            return y_hat;
        }

        inline af::array GaussianMixture::predict_proba(af::array & X)
        {
            int n_samples = X.dims(0), n_features = X.dims(1);
            af::array gamma(n_samples, this->n_components_);
            for(int j = 0; j < n_samples; ++j)
            {
                for(int k = 0; k < this->n_components_; ++k)
                {
                    af::array z = af::batchFunc(X.row(j), this->means_.row(k), utils::bsub);
                    gamma(j, k) = this->weights_.row(k) * af::exp(-af::matmul(z, af::inverse(this->covariances_[k]), z.T()) / 2.0) / (sqrt(pow(2 * M_PI, n_features) * abs(af::det<float>(this->covariances_[k]))));
                }
            }
            gamma = af::batchFunc(gamma, af::sum(gamma, 1), utils::bdiv);

            return gamma;
        }

        inline af::array GaussianMixture::fit_predict(af::array & X){
            this->fit(X);
            return this->predict(X);
        }
    }
}
#endif