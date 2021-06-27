#ifndef INSNET_OPERATOR_DEF_H
#define INSNET_OPERATOR_DEF_H

#include "insnet/base/def.h"
#include <cmath>

namespace insnet {

inline dtype fexp(const dtype& x) {
    return std::exp(x);
}

inline dtype flog(const dtype& x) {
    return std::log(x);
}

inline dtype dequal(const dtype& x, const dtype& y) {
    return 1;
}

inline dtype dtanh(dtype y) {
    return (1 + y) * (1 - y);
}

inline dtype dleaky_relu(const dtype& x, const dtype& y) {
    if (x < 0) return 0.1;
    return 1;
}

inline dtype dselu(const dtype& x, const dtype& y) {
    dtype lambda = 1.0507009873554804934193349852946;
    dtype alpha = 1.6732632423543772848170429916717;
    if (x <= 0) return lambda * alpha + y;
    return lambda;
}

inline dtype dsigmoid(dtype y) {
    return (1 - y) * y;
}

inline dtype dexp(const dtype& x, const dtype& y) {
    return y;
}

inline dtype drelu(dtype y) {
    return y > 0 ? 1 : 0;
}

inline dtype dlog(const dtype& x, const dtype& y) {
    if(x < 0.001) return 1000;
    return 1.0 / x;
}

inline dtype dsqrt(dtype y) {
    return 0.5 / y;
}

inline dtype fequal(const dtype& x) {
    return x;
}

inline dtype ftanh(const dtype& x) {
    return std::tanh(x);
}

inline dtype fsigmoid(const dtype& x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline dtype frelu(const dtype& x) {
    if (x <= 0) return 0;
    return x;
}

inline dtype fleaky_relu(const dtype& x) {
    if (x < 0) return (0.1*x);
    return x;
}

inline dtype fselu(const dtype& x) {
    dtype lambda = 1.0507009873554804934193349852946;
    dtype alpha = 1.6732632423543772848170429916717;
    if (x <= 0) return lambda * alpha * (std::exp(x) - 1);
    return lambda * x;
}

inline dtype fsqrt(const dtype &x) {
    return std::sqrt(x);
}

}

#endif
