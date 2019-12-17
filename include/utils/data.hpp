#ifndef MLKIT_DATA_HPP
#define MLKIT_DATA_HPP

namespace mk
{
	namespace utils
	{
		template<typename T>
		af::array load_dataset(const std::string& path)
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
	}
}

#endif