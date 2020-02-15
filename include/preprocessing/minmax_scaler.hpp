#ifndef MLKIT_MINMAX_SCALER
#define MLKIT_MINMAX_SCALER
namespace mk
{
    namespace preprocessing
    {
        class MinMaxScaler
        {
        private:
            std::pair<float, float> feature_range_;
        public:
            af::array min_;
            af::array scale_;
            af::array data_min_;
            af::array data_max_;
            af::array data_range_;
            std::vector<bool> flag_;

            MinMaxScaler(std::pair<float, float> feature_range=std::make_pair(0, 1));
            void fit(af::array &X);
            af::array transform(af::array &X);
            af::array fit_transform(af::array &X);
        };

        inline MinMaxScaler::MinMaxScaler(std::pair<float, float> feature_range):feature_range_(feature_range)
        {

        }

        inline void MinMaxScaler::fit(af::array &X)
        {
            this->data_min_ = af::min(X, 0);
            this->data_max_ = af::max(X, 0);
            this->data_range_ = this->data_max_ - this->data_min_;
            this->scale_ = (this->feature_range_.second - this->feature_range_.first) / this->data_range_;
            this->min_ = this->feature_range_.first - this->data_min_ * this->scale_;

            for(int i = 0; i < this->data_range_.dims(1); ++i)
            {
                this->flag_.push_back(af::allTrue<bool>(this->data_range_.col(i)));
            }
        }

        inline af::array MinMaxScaler::transform(af::array &X)
        {
            af::array X_copy = X;
            for(int i = 0; i < X.dims(1); ++i){
                if(this->flag_[i])
                    X_copy.col(i) = batchFunc(batchFunc(X_copy.col(i), this->scale_.col(i), utils::bmul), this->min_.col(i), utils::badd);
            }
            return X_copy;
        }

        inline af::array MinMaxScaler::fit_transform(af::array &X){
            this->fit(X);
            return this->transform(X);
        }

    }
}


#endif