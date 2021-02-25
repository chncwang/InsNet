#ifndef N3LDG_DEF_H
#define N3LDG_DEF_H

#include "NRMat.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>

#if USE_DOUBLE
#define USE_FLOAT 0
#else
#define USE_FLOAT 1
#endif

using namespace Eigen;

#if USE_FLOAT
typedef float dtype;
typedef Eigen::TensorMap<Eigen::Tensor<float, 1>>  Vec;
typedef Eigen::Map<Matrix<float, Dynamic, Dynamic, ColMajor> > Mat;
typedef MatrixXf MatrixXdtype;
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


#endif
