#ifndef MLKIT_StandardScaler_HPP
#define MLKIT_StandardScaler_HPP
namespace mk
{
    namespace preprocessing
    {
        class StandardScaler
        {
            public:
                af::array scale_;
                af::array mean_;
                af::array var_;
                af::array flag_;

                StandardScaler(bool with_mean=true, bool with_std=true);

                void fit(af::array & X);
                af::array transform(af::array & X);
                af::array fit_transform(af::array & X);
            private:
                bool with_mean_, with_std_;
        };

        inline StandardScaler::StandardScaler(bool with_mean, bool with_std): with_mean_(with_mean), with_std_(with_std)
        {

        }

        inline void StandardScaler::fit(af::array & X)
        {
            if(this->with_mean_)
                this->mean_ = af::mean(X, 0);
            if(this->with_std_){
                this->var_ = af::var(X, false, 0);
                this->flag_ = !af::iszero(this->var_);
                this->scale_ = af::sqrt(this->var_);
            }
        }

        inline af::array StandardScaler::transform(af::array & X)
        {
            af::array X_copy = X;
            for(int i = 0; i < X.dims(1); ++i)
            {
                if(this->flag_(af::span, i).scalar<char>())
                {
                    if(this->with_mean_)
                        X_copy(af::span, i) = batchFunc(X_copy(af::span, i), this->mean_(af::span, i), utils::bsub);
                    if(this->with_std_)
                        X_copy(af::span, i) = batchFunc(X_copy(af::span, i), this->scale_(af::span, i), utils::bdiv);
                }
            }

            return X_copy;
        }

        inline af::array StandardScaler::fit_transform(af::array & X)
        {
            this->fit(X);
            return this->transform(X);
        }
    }
}

#endif