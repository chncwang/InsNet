/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Eigen/Dense"
#include "BaseParam.h"

// Notice: aux is an auxiliary variable to help parameter updating
class Param : public BaseParam {
  public:
    Tensor2D aux_square;
    Tensor2D aux_mean;
    int iter;

    // allow sparse and dense parameters have different parameter initialization methods
    inline void initial(int outDim, int inDim) {
        val.init(outDim, inDim);
        grad.init(outDim, inDim);
        aux_square.init(outDim, inDim);
        aux_mean.init(outDim, inDim);

        dtype bound = sqrt(6.0 / (outDim + inDim + 1));
        val.random(bound);
        iter = 0;
    }

    inline int outDim() {
        return val.row;
    }

    inline int inDim() {
        return val.col;
    }

    inline void clearGrad() {
        grad.zero();
    }

    inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_square.vec() = aux_square.vec() + grad.vec().square();
        val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
    }

    inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
        iter++;
    }

    inline void randpoint(int& idx, int &idy) {
        //select indexes randomly
        std::vector<int> idRows, idCols;
        idRows.clear();
        idCols.clear();
        for (int i = 0; i < val.row; i++)
            idRows.push_back(i);
        for (int i = 0; i < val.col; i++)
            idCols.push_back(i);

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idy = idRows[0];
        idx = idCols[0];
    }

    inline dtype squareGradNorm() {
        dtype sumNorm = 0.0;
        for (int i = 0; i < grad.size; i++) {
            sumNorm += grad.v[i] * grad.v[i];
        }
        return sumNorm;
    }

    inline void rescaleGrad(dtype scale) {
        grad.vec() = grad.vec() * scale;
    }

    inline void save(std::ofstream &os)const {
        val.save(os);
        aux_square.save(os);
        aux_mean.save(os);
        os << iter << endl;
    }

    inline void load(std::ifstream &is) {
        val.load(is);
        aux_square.load(is);
        aux_mean.load(is);
        is >> iter;
    }
};

#endif /* PARAM_H_ */
