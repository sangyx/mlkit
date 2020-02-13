#ifndef MLKIT_KMEANS_HPP
#define MLKIT_KMEANS_HPP
namespace mk
{
    namespace cluster
    {
        class KMeans
        {
        private:
            int n_clusters_, n_init_, max_iter_, random_state_;
            std::string init_;
            float tol_;
        public:
            af::array cluster_centers_;
            af::array labels_;
            float inertia_;
            int n_iter_;

            KMeans(int n_clusters=8, std::string init="k-means++", int n_init=10, int max_iter=300, float tol=0.0001, int random_state=-1);

            void fit(af::array & X);
            af::array fit_predict(af::array & X);
            af::array fit_transform(af::array & X);
            af::array transform(af::array & X);
        };

        inline KMeans::KMeans(int n_clusters, std::string init, int n_init, int max_iter, float tol, int random_state): n_clusters_(n_clusters), init_(init), n_init_(n_init), max_iter_(max_iter), tol_(tol), random_state_(random_state)
        {
        }

        inline void KMeans::fit(af::array & X)
        {
            std::unordered_map<int, int> cluster_count;
            af::array X_dist = utils::minkowski_distance(X, X);
            af::array cluster_centers, cluster_dist, inertia_dist;
            std::vector<int> indices;

            std::tie(cluster_centers, indices) = utils::sample(X, this->n_clusters_);

            if(this->init_ == "k-means++")
            {
                for(int i = 0; i < n_init_; ++i)
                {
                    cluster_dist = utils::choice(X_dist, indices, 1);
                    af::array paf = af::sum(cluster_dist, 1) / af::sum<float>(cluster_dist, 1);
                    std::vector<float> p = utils::array2vec(paf);
                    std::tie(cluster_centers, indices) = utils::sample(X, this->n_clusters_, p);
                }
            }

            this->inertia_ = af::sum<float>(cluster_dist);
            cluster_dist = utils::choice(X_dist, indices, 1);
            af::min(inertia_dist, this->labels_, cluster_dist, 1);

            for(this->n_iter_ = 0; this->n_iter_ < this->max_iter_; ++this->n_iter_)
            {
                float inertia = af::sum<float>(inertia_dist);

                if(this->inertia_ - inertia < this->tol_)
                    break;

                this->inertia_ = inertia;
                cluster_centers = 0;
                cluster_count.clear();
                for(int i = 0; i < X.dims(0); ++i)
                {
                    int label = this->labels_.row(i).scalar<unsigned int>();
                    cluster_centers.row(label) += X.row(i);
                    cluster_count[label]++;
                }

                for(int k = 0; k < this->n_clusters_; ++k)
                {
                    if(cluster_count[k])
                        cluster_centers.row(k) /= cluster_count[k];
                }

                cluster_dist = utils::minkowski_distance(X, cluster_centers);
                af::min(inertia_dist, this->labels_, cluster_dist, 1);
            }
            this->inertia_ = af::sum<float>(inertia_dist);
            this->cluster_centers_ = cluster_centers;
        }

        inline af::array KMeans::fit_predict(af::array & X)
        {
            this->fit(X);
            return this->labels_;
        }

        inline af::array KMeans::transform(af::array & X)
        {
            return utils::minkowski_distance(X, this->cluster_centers_);
        }

        inline af::array KMeans::fit_transform(af::array & X)
        {
            this->fit(X);
            return this->transform(X);
        }

    }
}

#endif