#ifndef MLKIT_HPP
#define MLKIT_HPP

#include <cmath>
#include <assert.h>
#include <random>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <numeric>

#include "arrayfire.h"

namespace mk {
	struct version
	{
		static const unsigned int major = 0;
		static const unsigned int minor = 1;
		static const unsigned int patch = 0;

		static inline std::string as_string()
		{

			std::stringstream ss;
			ss << version::major << '.' << version::minor << '.' << version::patch;

			return ss.str();
		}
	};
}

#ifdef _MSC_VER
#pragma warning(disable : 4018)
#endif

#include "utils/data.hpp"
#include "utils/broadcasting.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/minmax_scaler.hpp"
#include "statistics/linear_model/linear_regression.hpp"
#include "statistics/linear_model/logistic_regression.hpp"
// #include "statistics/neighbors/kneighbors_classifier.hpp"
#endif