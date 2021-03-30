#ifndef N3LDG_DEF_H
#define N3LDG_DEF_H

#include "eigen/Eigen/Dense"
#include "eigen//unsupported/Eigen/CXX11/Tensor"

#if USE_DOUBLE
#define USE_FLOAT 0
#else
#define USE_FLOAT 1
#endif

namespace n3ldg_plus {

#if USE_FLOAT
typedef float dtype;
typedef Eigen::TensorMap<Eigen::Tensor<float, 1>>  Vec;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > Mat;
typedef Eigen::MatrixXf MatrixXdtype;
#else
typedef double dtype;
typedef Eigen::TensorMap<Eigen::Tensor<double, 1>>  Vec;
typedef Eigen::Map<Matrix<double, Dynamic, Dynamic, ColMajor> > Mat;
typedef Matrix<double, Dynamic, Dynamic> MatrixXdtype;
#endif

constexpr dtype INF = 1e30;
enum ActivatedEnum {
    EXP = 0,
    TANH = 1,
    SIGMOID = 2,
    RELU = 3,
    LEAKY_RELU = 4,
    SELU = 5,
    SQRT=6
};

}

#endif
