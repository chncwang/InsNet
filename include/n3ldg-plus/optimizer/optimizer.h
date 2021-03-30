#ifndef ModelUpdate_H_
#define ModelUpdate_H_

#include "n3ldg-plus/param/base-param.h"
#include "n3ldg-plus/util/util.h"

namespace n3ldg_plus {

class ModelUpdate {
public:
    std::vector<BaseParam*> _params;

    dtype _reg, _alpha, _eps;
    dtype _belta1, _belta2;

    ModelUpdate() {
        _params.clear();

        _reg = 1e-8;
        _alpha = 0.001;
        _eps = 1e-8;


        _belta1 = 0.9;
        _belta2 = 0.999;
    }

    void addParam(BaseParam* param) {
        for (BaseParam * p : _params) {
            if (p == param) {
                abort();
            }
        }
        _params.push_back(param);
    }

    void setParams(const std::vector<BaseParam *> &params) {
        std::cout << "setParams - params size:" << params.size() << std::endl;
        _params = params;
    }

    void update() {
        for (int idx = 0; idx < _params.size(); idx++) {
            _params[idx]->updateAdagrad(_alpha, _reg, _eps);
            _params[idx]->clearGrad();
        }
    }

    void update(dtype maxScale) {
        dtype sumNorm = 0.0;
        for (int idx = 0; idx < _params.size(); idx++) {
            sumNorm += _params[idx]->squareGradNorm();
        }
        dtype norm = sqrt(sumNorm);
        if (norm > maxScale) {
            dtype scale = maxScale / norm;
            for (int idx = 0; idx < _params.size(); idx++) {
                _params[idx]->rescaleGrad(scale);
            }
        }

        update();
    }

    void updateAdam() {
        for (int idx = 0; idx < _params.size(); idx++) {
            _params[idx]->updateAdam(_belta1, _belta2, _alpha, _reg, _eps);
            _params[idx]->clearGrad();
        }
    }

    void updateAdam(dtype maxScale) {
#if TEST_CUDA
        maxScale = 10;
#endif
        dtype sumNorm = 0.0;
        for (int idx = 0; idx < _params.size(); idx++) {
            sumNorm += _params[idx]->squareGradNorm();
        }
        dtype norm = sqrt(sumNorm);
        if (maxScale > 0 && norm > maxScale) {
            dtype scale = maxScale / norm;
            for (int idx = 0; idx < _params.size(); idx++) {
                _params[idx]->rescaleGrad(scale);
            }
        }

        updateAdam();
#if TEST_CUDA
        for (BaseParam *p : _params) {
            p->copyFromHostToDevice();
        }
#endif
    }

    void updateAdamW(dtype max_scale) {
#if TEST_CUDA
        max_scale = 10;
#endif
        dtype sumNorm = 0.0;
        for (int idx = 0; idx < _params.size(); idx++) {
            sumNorm += _params[idx]->squareGradNorm();
        }
        dtype norm = sqrt(sumNorm);
        if (max_scale > 0 && norm > max_scale) {
            dtype scale = max_scale / norm;
            for (int idx = 0; idx < _params.size(); idx++) {
                _params[idx]->rescaleGrad(scale);
            }
        }

        updateAdamW();
#if TEST_CUDA
        for (BaseParam *p : _params) {
            p->copyFromHostToDevice();
        }
#endif
    }

    void updateAdamW() {
        for (int idx = 0; idx < _params.size(); idx++) {
            _params[idx]->updateAdamW(_belta1, _belta2, _alpha, _reg, _eps);
            _params[idx]->clearGrad();
        }
    }

    void rescaleGrad(dtype scale) {
        for (int idx = 0; idx < _params.size(); idx++) {
            _params[idx]->rescaleGrad(scale);
        }
    }

    void clearGrad() {
        for (int idx = 0; idx < _params.size(); idx++) {
            _params[idx]->clearGrad();
        }
    }

    void gradClip(dtype maxScale) {
        dtype sumNorm = 0.0;
        for (int idx = 0; idx < _params.size(); idx++) {
            sumNorm += _params[idx]->squareGradNorm();
        }
        if (std::isnan(double(sumNorm)) || sumNorm > 1e20) { //too large
            clearGrad();
            return;
        }
        dtype norm = sqrt(sumNorm);
        if (maxScale > 0 && norm > maxScale) {
            dtype scale = maxScale / norm;
            for (int idx = 0; idx < _params.size(); idx++) {
                _params[idx]->rescaleGrad(scale);
            }
        }
    }

    void clear() {
        _params.clear();
    }
};

}

#endif /* ModelUpdate_H_ */
