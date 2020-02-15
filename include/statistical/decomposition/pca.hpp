#ifndef MLKIT_PCA_HPP
#define MLKIT_PCA_HPP
namespace mk
{
    namespace decomposition
    {
        class PCA
        {
        public:
            af::array components_, explained_variance_, explained_variance_ratio_, mean_;
            int n_components_, n_features_, n_samples_;
            PCA(int n_components=3);
            void fit(af::array & X);
            af::array transform(af::array & X);
            af::array fit_transform(af::array & X);
        };

        inline PCA::PCA(int n_components): n_components_(n_components)
        {
        }

        inline void PCA::fit(af::array & X)
        {
            this->n_samples_ = X.dims(0);
            this->n_features_ = X.dims(1);
            this->mean_ = af::mean(X);
            af::array X_norm = batchFunc(X, this->mean_, utils::bsub);
            af::array u, s, vt;
            af::svd(u, s, vt, X_norm);
            this->components_ = vt(af::seq(this->n_components_), af::span);
            this->explained_variance_ = s(af::seq(this->n_components_), af::span);
            this->explained_variance_ratio_ = this->explained_variance_ / af::sum<float>(s);
        }

        inline af::array PCA::transform(af::array & X)
        {
            af::array X_norm = batchFunc(X, this->mean_, utils::bsub);
            return af::matmul(X_norm, this->components_.T());
        }

        inline af::array PCA::fit_transform(af::array & X)
        {
            this->fit(X);
            return this->transform(X);
        }

    }
}

#endif