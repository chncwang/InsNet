#ifndef INSNET_OPTIMIZER_H
#define INSNET_OPTIMIZER_H

#include "insnet/param/base-param.h"

namespace insnet {

class Optimizer {
public:
    Optimizer(const std::vector<BaseParam *> &params, dtype learning_rate) : params_(params),
    lr_(learning_rate) {}

    virtual void optimize() = 0;

    void step();

    void step(dtype clip_value) {
        clipGrad(clip_value);
        step();
    }

    void setLearningRate(dtype learning_rate) {
        lr_ = learning_rate;
    }

    dtype getLearningRate() const {
        return lr_;
    }

protected:
    std::vector<BaseParam *> params_;
    dtype lr_;

private:
    void clipGrad(dtype clip_value);
};

}

#endif
