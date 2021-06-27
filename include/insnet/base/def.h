#ifndef INSNET_DEF_H
#define INSNET_DEF_H

#include <string>

#if USE_DOUBLE
#define USE_FLOAT 0
#else
#define USE_FLOAT 1
#endif

namespace insnet {

#if USE_FLOAT
typedef float dtype;
#else
typedef double dtype;
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

extern const std::string UNKNOWN_WORD;
}

#endif
