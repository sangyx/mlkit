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
	}
}

#endif