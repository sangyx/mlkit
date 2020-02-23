#ifndef MLKIT_DECISION_TREE_CLASSIFIER_HPP
#define MLKIT_DECISION_TREE_CLASSIFIER_HPP
namespace mk
{
    namespace tree
    {
        struct Tree {
        int feature;
        float split;
        int label;
        Tree *left;
        Tree *right;
        Tree(int feature, float split) : feature(feature), split(split), left(NULL), right(NULL), label(-1) {}
        Tree(int label): label(label), left(NULL), right(NULL) {}
    };


        class DecisionTreeClassifier
        {
        private:
            int min_samples_split_;
            float min_impurity_decrease_;
            Tree* gen_tree(af::array & X, af::array & y);
            float cal_gini(af::array & y, std::vector<int> samples);
        public:
            af::array feature_importances_;
            int n_features_, n_samples;
            Tree *tree_;

            DecisionTreeClassifier(int min_samples_split=2, float min_impurity_decrease=0.0);
            ~DecisionTreeClassifier();
            void fit(af::array & X, af::array & y);
            af::array predict(af::array & X);
        };

        inline DecisionTreeClassifier::DecisionTreeClassifier(int min_samples_split, float min_impurity_decrease): min_samples_split_(min_samples_split), min_impurity_decrease_(min_impurity_decrease)
        {
        }

        inline DecisionTreeClassifier::~DecisionTreeClassifier()
        {
            delete this->tree_;
        }

        inline float DecisionTreeClassifier::cal_gini(af::array & y, std::vector<int> samples)
        {
            std::unordered_map<float, int> count;
            for(int idx: samples)
            {
                count[y.row(idx).scalar<float>()]++;
            }

            float gini = 1;
            for(auto it = count.begin(); it != count.end(); ++it)
            {
                gini -= it->second * it->second;
            }
            return gini;
        }

        inline Tree* DecisionTreeClassifier::gen_tree(af::array & X, af::array & y)
        {
            int n_samples = X.dims(0);
            af::array labels = af::setUnique(y, true);
            if(n_samples <= this->min_samples_split_ || labels.elements() == 1){
                Tree *root = new Tree(labels(0).scalar<float>());
                return root;
            }
            int feature;
            float split, gini = FLT_MAX;
            std::vector<int> left, right;
            for(int i = 0; i < this->n_features_; ++i)
            {
                af::array vals = af::setUnique(X.col(i), true);
                if(vals.elements() < 2)
                    continue;
                for(int j = 0; j < vals.elements() - 1; ++j)
                {
                    float cur_split = (vals(j) + vals(j+1)).scalar<float>() / 2;
                    std::vector<int> cur_left, cur_right;
                    for(int k= 0; k < n_samples; ++k)
                    {
                        if(X(k, i).scalar<float>() <= cur_split)
                            cur_left.push_back(k);
                        else
                            cur_right.push_back(k);
                    }
                    if(n_samples > 2 * this->min_samples_split_ && (cur_left.size() <= std::max(0.2 * n_samples, 1.0) || cur_right.size() <=  std::max(0.2 * n_samples, 1.0) ))
                        continue;

                    float cur_gini = (cur_left.size() * cal_gini(y, cur_left) + cur_right.size() * cal_gini(y, cur_right)) / n_samples;
                    if(cur_gini < gini)
                    {
                        gini = cur_gini;
                        feature = i;
                        split = cur_split;
                        left = cur_left;
                        right = cur_right;
                    }
                    if(cur_gini == 0)
                        break;
                }
            }

            Tree *root = new Tree(feature, split);
            af::array X_left = utils::choice(X, left), X_right = utils::choice(X, right);
            af::array y_left = utils::choice(y, left), y_right = utils::choice(y, right);

            root->left = gen_tree(X_left, y_left);
            root->right = gen_tree(X_right, y_right);
            return root;
        }

        inline void DecisionTreeClassifier::fit(af::array & X, af::array & y)
        {
            this->n_samples = X.dims(0), this->n_features_ = X.dims(1);
            this->tree_ = gen_tree(X, y);
        }

        inline af::array DecisionTreeClassifier::predict(af::array & X)
        {
            int n_sample = X.dims(0);
            af::array y_hat(n_sample, 1);
            for(int i = 0; i < n_sample; ++i)
            {
                Tree* p = this->tree_;
                while(p->left != NULL && p->right != NULL)
                {
                    if(X(i, p->feature).scalar<float>() <= p->split)
                        p = p->left;
                    else
                        p = p->right;
                }
                y_hat.row(i) = p->label;
            }
            return y_hat;
        }

    }
}
#endif