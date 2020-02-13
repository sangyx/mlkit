#ifndef MLKIT_DATA_HPP
#define MLKIT_DATA_HPP

namespace mk
{
	namespace utils
	{
		template<typename T>
		inline af::array load_dataset(const std::string& path)
		{
			std::ifstream indata(path);
			if(!indata.is_open())
			{
				throw "file can't open!";
			}

			int rows = 0, cols = 0;
			std::string line;
			T tmp;

			while (std::getline(indata, line))
			{
				if (rows == 0)
				{
					std::istringstream iss(line);
					while (iss >> tmp)
						cols++;
				}
				line.clear();
				rows++;
			}

			indata.clear();
			indata.seekg(0);

			af::array data(rows, cols);

			for(int row = 0; row < rows; ++row)
			{
				for(int col = 0; col < cols; ++col)
				{
					indata >> tmp;
					data(row, col) = tmp;
				}
			}
			indata.close();

			std::cout << "load data completed! data shape: (" << rows << "," << cols << ")" << std::endl;
			return data;
		}

		inline std::tuple<af::array, af::array, af::array, af::array> train_test_split(af::array & X, af::array & y, float train_size, int random_state=-1, bool shuffle=true)
		{
			int m = X.dims(0), n = X.dims(1), p = y.dims(1);
			int train_num = ceil(m * train_size), test_num = m - train_num;
			std::vector<int> idx(m);
			std::iota(idx.begin(), idx.end(), 0);

			if(shuffle)
				std::shuffle(idx.begin(), idx.end(), std::default_random_engine(random_state));

			af::array X_train(train_num, n), y_train(train_num, p), X_test(test_num, n), y_test(test_num, p);
			for(int i = 0; i < train_num; ++i)
			{
				X_train.row(i) = X.row(idx[i]);
				y_train.row(i) = y.row(idx[i]);
			}

			for(int i = train_num; i < m; ++i)
			{
				X_test.row(i-train_num) = X.row(idx[i]);
				y_test.row(i-train_num) = y.row(idx[i]);
			}

			return std::make_tuple(X_train, y_train, X_test, y_test);
		}

		inline std::tuple<af::array, std::vector<int>> sample(af::array & X, int num, int random_state=-1)
		{
			int m = X.dims(0), n = X.dims(1);
			af::array X_sample(num, n);
			std::vector<int> idx(m), indices;
			std::iota(idx.begin(), idx.end(), 0);

			std::shuffle(idx.begin(), idx.end(), std::default_random_engine(random_state));

			for(int i = 0; i < num; ++i)
			{
				X_sample.row(i) = X.row(idx[i]);
				indices.push_back(idx[i]);
			}
			return std::make_tuple(X_sample, indices);
		}

		inline std::tuple<af::array, std::vector<int>> sample(af::array & X, int num, std::vector<float> p, int random_state=-1)
		{
			int m = X.dims(0), n = X.dims(1), i = 0;
			af::array X_sample(num, n);
			std::vector<int> indices;
			std::unordered_set<int> idx;
			std::mt19937 gen(random_state);
			std::discrete_distribution<> d{p.begin(), p.end()};
			while(idx.size() < num)
			{
				idx.insert(d(gen));
			}

			for(auto it = idx.begin(); it != idx.end(); ++it)
			{
				X_sample.row(i) = X.row(*it);
				indices.push_back(*it);
				++i;
			}

			return std::make_tuple(X_sample, indices);

		}

		inline af::array choice(af::array & X, std::vector<int>& idx, int dim=0)
		{
			af::array X_copy;
			if(dim == 1)
				X_copy = X.T();
			else
				X_copy = X;
			int m = idx.size(), n = X_copy.dims(1);
			af::array X_choice(m, n);
			for(int i = 0; i < m; ++i)
			{
				X_choice.row(i) = X_copy.row(idx[i]);
			}

			if(dim == 1)
				return X_choice.T();
			return X_choice;
		}

		inline std::vector<float> array2vec(af::array & X)
		{
			int m = X.dims(0), n = X.dims(1);
			std::vector<float> vec;
			for(int i = 0; i < m; ++i)
			{
				for(int j = 0; j < n; ++j)
					vec.push_back(X(i, j).scalar<float>());
			}

			return vec;
		}
	}
}

#endif