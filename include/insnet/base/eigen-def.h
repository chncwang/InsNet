#ifndef INSNET_EIGEN_DEF_H
#define INSNET_EIGEN_DEF_H

#include "eigen/Eigen/Dense"
#include "eigen//unsupported/Eigen/CXX11/Tensor"
#include "insnet/base/def.h"

namespace insnet {

#if USE_FLOAT
typedef Eigen::TensorMap<Eigen::Tensor<float, 1>>  Vec;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > Mat;
typedef Eigen::MatrixXf MatrixXdtype;
#else
typedef Eigen::TensorMap<Eigen::Tensor<double, 1>>  Vec;
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXdtype;
#endif

}

#endif
