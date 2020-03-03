#ifndef MLKIT_LINEAR_SVC_HPP
#define MLKIT_LINEAR_SVC_HPP
namespace mk
{
    namespace svm
    {
        class LinearSVC
        {
        private:
            double tol_, C_;
            int max_iter_;
        public:
            af::array sv_X_, sv_y_;
            af::array alpha_;
            double intercept_;
            int n_iter_;
            LinearSVC(double tol=0.0001, double C=1.0, int max_iter=1000);
            void fit(af::array & X, af::array & y);
            af::array predict(af::array & X);
        };

        inline LinearSVC::LinearSVC(double tol, double C, int max_iter): tol_(tol), C_(C), max_iter_(max_iter)
        {
        }

        inline void LinearSVC::fit(af::array & X, af::array & y)
        {
            int n_samples = X.dims(0), n_features = X.dims(1);
            af::array alpha = af::constant(0, n_samples);
            std::vector<double> E(n_samples, 0);
            af::array K = af::matmul(X, X.T());
            for(this->n_iter_ = 0; this->n_iter_ < this->max_iter_; ++this->n_iter_)
            {
                for(int i = 0; i < n_samples; ++i)
                {
                    double gxi = af::sum<float>(alpha * y * K.col(i)) + this->intercept_;
                    double yi = y.row(i).scalar<float>();
                    double ai = alpha.row(i).scalar<float>();

                    if(abs(ai) < this->tol_ && yi * gxi >= 1)
                        continue;
                    if(abs(ai - this->C_) < this->tol_ && yi * gxi <= 1)
                        continue;
                    if(ai > -this->tol_ && ai < this->C_ + this->tol_ && abs(yi * gxi - 1) < this->tol_)
                        continue;

                    double E1 = gxi - yi;
                    double E2 = 0, maxE1_E2 = -1;
                    int maxJ = -1;
                    for(int j = 0; j < n_samples; ++j)
                    {
                        if(E[j] != 0){
                            double E2_tmp = af::sum<float>(alpha * y * K.col(j)) + this->intercept_ - y.row(j).scalar<float>();
                            if(abs(E1 - E2_tmp) > maxE1_E2)
                            {
                                maxE1_E2 = abs(E1 - E2_tmp);
                                E2 = E2_tmp;
                                maxJ = j;
                            }
                        }
                    }
                    if(maxJ == -1)
                    {
                        maxJ = i;
                        while(maxJ == i)
                            maxJ = rand() % n_samples;
                        E2 = af::sum<float>(alpha * y * K.col(maxJ)) + this->intercept_ - y.row(maxJ).scalar<float>();
                    }

                    int y1 = y.row(i).scalar<float>(), y2 = y.row(maxJ).scalar<float>();
                    double alpha_old_1 = ai, alpha_old_2 = alpha.row(maxJ).scalar<float>();

                    double L = 0, H = 0;
                    if(y1 != y2){
                        L = std::max(0.0, alpha_old_2 - alpha_old_1);
                        H = std::min(this->C_, this->C_ + alpha_old_2 - alpha_old_1);
                    }else{
                        L = std::max(0.0, alpha_old_2 + alpha_old_1 - this->C_);
                        H = std::min(this->C_, alpha_old_2 + alpha_old_1);
                    }

                    if(L == H)
                        continue;
                    double k11 = K(i, i).scalar<float>(), k22 = K(maxJ, maxJ).scalar<float>(), k21 = K(maxJ, i).scalar<float>(), k12 = K(i, maxJ).scalar<float>();
                    double alpha_new_2 = alpha_old_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12);
                    if(alpha_new_2 < L)
                        alpha_new_2 = L;
                    else if(alpha_new_2 > H)
                        alpha_new_2 = H;
                    double alpha_new_1 = alpha_old_1 + y1 * y2 * (alpha_old_2 - alpha_new_2);
                    double b1_new = -1 * E1 - y1 * k11 * (alpha_new_1 - alpha_old_1) - y2 * k21 * (alpha_new_2 - alpha_old_2) + this->intercept_;
                    double b2_new = -1 * E2 - y1 * k12 * (alpha_new_1 - alpha_old_1) - y2 * k22 * (alpha_new_2 - alpha_old_2) + this->intercept_;

                    if(alpha_new_1 > 0 && alpha_new_1 < this->C_)
                        this->intercept_ = b1_new;
                    else if(alpha_new_2 > 0 && alpha_new_2 < this->C_)
                        this->intercept_ = b2_new;
                    else
                        this->intercept_ = (b1_new + b2_new) / 2;

                    alpha.row(i) = alpha_new_1;
                    alpha.row(maxJ) = alpha_new_2;
                    E[i] = af::sum<float>(alpha * y * K.col(i)) + this->intercept_ - y.row(i).scalar<float>();
                    E[maxJ] = af::sum<float>(alpha * y * K.col(maxJ)) + this->intercept_ - y.row(maxJ).scalar<float>();

                    if(abs(alpha_new_2 - alpha_old_2) <= this->tol_)
                        break;

                }
            }

            std::vector<int> idx;
            for(int i = 0; i < n_samples; ++i)
            {
                if(alpha.row(i).scalar<float>() > 0)
                    idx.push_back(i);
            }

            this->sv_X_ = utils::choice(X, idx);
            this->sv_y_ = utils::choice(y, idx);
            this->alpha_ = utils::choice(alpha, idx);
        }

        inline af::array LinearSVC::predict(af::array & X)
        {
            int n_samples = X.dims(0);
            af::array y_hat(n_samples);
            for(int i = 0; i < n_samples; ++i)
            {
                af_print(this->alpha_ * this->sv_y_ * af::matmul(this->sv_X_, X.row(i).T()));
                double label = af::sum<float>(this->alpha_ * this->sv_y_ * af::matmul(this->sv_X_, X.row(i).T())) + this->intercept_;
                if(label > 0)
                    y_hat.row(i) = 1;
                else if(label < 0)
                    y_hat.row(i) = -1;
                else
                    y_hat.row(i) = 0;
            }

            return y_hat;
        }
    }
}
#endif