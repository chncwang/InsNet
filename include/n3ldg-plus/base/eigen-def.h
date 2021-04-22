#ifndef N3LDG_PLUS_EIGEN_DEF_H
#define N3LDG_PLUS_EIGEN_DEF_H

#include "eigen/Eigen/Dense"
#include "eigen//unsupported/Eigen/CXX11/Tensor"
#include "n3ldg-plus/base/def.h"

namespace n3ldg_plus {

#if USE_FLOAT
typedef Eigen::TensorMap<Eigen::Tensor<float, 1>>  Vec;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > Mat;
typedef Eigen::MatrixXf MatrixXdtype;
#else
typedef Eigen::TensorMap<Eigen::Tensor<double, 1>>  Vec;
typedef Eigen::Map<Matrix<double, Dynamic, Dynamic, ColMajor> > Mat;
typedef Matrix<double, Dynamic, Dynamic> MatrixXdtype;
#endif

}

#endif
