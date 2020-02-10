#ifndef MLKIT_KNEIGHBORS_CLASSIFIER_HPP
#define MLKIT_KNEIGHBORS_CLASSIFIER_HPP
namespace mk{
    namespace neighbors{
        class KNeighborsClassifier
        {
        private:
            int n_neighbors_, p_;
            af::array X_, y_;
        public:
            KNeighborsClassifier(int n_neighbors=5, int p=2);
            void fit(af::array & X, af::array & y);
            af::array predict(af::array & X);
            float score(af::array & X, af::array & y);
        };

        inline KNeighborsClassifier::KNeighborsClassifier(int n_neighbors, int p): n_neighbors_(n_neighbors), p_(p)
        {
        }

        inline void KNeighborsClassifier::fit(af::array & X, af::array & y)
        {
            this->X_ = X;
            this->y_ = y;
        }

        inline af::array KNeighborsClassifier::predict(af::array & X)
        {
            int m = X.dims(0);
            af::array y(m, 1), dist;
            dist = utils::minkowski_distance(X, this->X_, this->p_);
            af::array values, indices;
            af::topk(values, indices, dist.T(), this->n_neighbors_, 0, AF_TOPK_MIN);
            for(int i = 0; i < m; ++i)
            {
                std::unordered_map<float, int> score;
                for(int j = 0; j < this->n_neighbors_; ++j)
                {
                    int idx = indices(j, i).scalar<unsigned int>();
                    float label = this->y_(idx, 0).scalar<float>();
                    if(score.find(label) == score.end())
                        score[label] = 1;
                    else
                        score[label]++;
                }
                std::vector<std::pair<float, int>> score_vec(score.begin(), score.end());
                std::sort(score_vec.begin(), score_vec.end(), [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs){
                    return lhs.second > rhs.second;
                });
                y(i, 0) = score_vec[0].first;
            }

            return y;
        }

        inline float KNeighborsClassifier::score(af::array & X, af::array & y)
        {
            af::array h = predict(X);
            float accuracy = 100 * af::count<float>(h == y) / h.elements();
            std::cout << "Accuracy: " << accuracy << "%" << std::endl;
            return accuracy;
        }
    }
}

#endif