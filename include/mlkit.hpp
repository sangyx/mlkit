#ifndef MLKIT_HPP
#define MLKIT_HPP

#include <cmath>
#include <assert.h>
#include <random>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
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
#include "utils/metrics.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/minmax_scaler.hpp"
#include "statistical/linear_model/linear_regression.hpp"
#include "statistical/linear_model/logistic_regression.hpp"
#include "statistical/neighbors/kneighbors_classifier.hpp"
#include "statistical/cluster/kmeans.hpp"
#include "statistical/decomposition/pca.hpp"
#endif