/*
 * APParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef AVGPARAM_H_
#define AVGPARAM_H_

#include "BaseParam.h"
#include "NRMat.h"
using namespace nr;

// Notice: aux is an auxiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
struct APParam : BaseParam {
    Tensor2D aux;
    NRVec<bool> indexers;
    int max_update;
    NRVec<int> last_update;

    // allow sparse and dense parameters have different parameter initialization methods
    inline void initial(int outDim, int inDim) {
        //not in the aligned memory pool
        val.init(outDim, inDim);
        grad.init(outDim, inDim);
        aux.init(outDim, inDim);
        indexers.resize(inDim);
        indexers = false;
        max_update = 0;
        last_update.resize(inDim);
        last_update = 0;
    }

    inline void clearGrad() {
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val.row; idx++) {
                grad[index][idx] = 0;
            }
        }
        indexers = false;
    }

    inline int outDim() {
        return val.row;
    }

    inline int inDim() {
        return val.col;
    }

    inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
        max_update++;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val.row; idx++) {
                aux[index][idx] += (max_update - last_update[index]) * val[index][idx] - grad[index][idx];
                val[index][idx] = val[index][idx] - grad[index][idx];
            }
            last_update[index] = max_update;
        }
    }

    inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
        max_update++;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val.row; idx++) {
                aux[index][idx] += (max_update - last_update[index]) * val[index][idx] - grad[index][idx];
                val[index][idx] = val[index][idx] - grad[index][idx];
            }
            last_update[index] = max_update;
        }
    }

    inline void randpoint(int& idx, int &idy) {
        //select indexes randomly
        std::vector<int> idRows, idCols;
        idRows.clear();
        idCols.clear();
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            idCols.push_back(index);
        }

        for (int i = 0; i < val.row; i++) {
            idRows.push_back(i);
        }

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idx = idCols[0];
        idy = idRows[0];
    }

    inline dtype squareGradNorm() {
        dtype sumNorm = 0.0;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val.row; idx++) {
                sumNorm += grad[index][idx] * grad[index][idx];
            }
        }

        return sumNorm;
    }

    inline void rescaleGrad(dtype scale) {
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val.row; idx++) {
                grad[index][idx] = grad[index][idx] * scale;
            }
        }
    }

    inline void sumWeight(int featId) {
        if (last_update[featId] < max_update) {
            int times = max_update - last_update[featId];
            for (int idx = 0; idx < val.row; idx++) {
                aux[featId][idx] += val[featId][idx] * times;
                last_update[featId] = max_update;
            }
        }
    }

    inline void value(const int& featId, Tensor1D& out, const bool& bTrain) {
        if (out.dim != val.row) {
            std::cout << "warning: output dim not equal lookup param dim." << std::endl;
        }
        if (bTrain) {
            for (int idx = 0; idx < val.row; idx++) {
                out[idx] = val[featId][idx];
            }
        } else {
            sumWeight(featId);
            for (int idx = 0; idx < val.row; idx++) {
                out[idx] = aux[featId][idx];
            }
        }
    }

    inline void value(const vector<int>& featIds, Tensor1D& out, const bool& bTrain) {
        if (out.dim != val.row) {
            std::cout << "warning: output dim not equal lookup param dim." << std::endl;
        }
        int featNum = featIds.size();
        int featId;
        if (bTrain) {
            for (int i = 0; i < featNum; i++) {
                featId = featIds[i];
                for (int idx = 0; idx < val.row; idx++) {
                    out[idx] += val[featId][idx];
                }
            }
        } else {
            for (int i = 0; i < featNum; i++) {
                featId = featIds[i];
                sumWeight(featId);
                for (int idx = 0; idx < val.row; idx++) {
                    out[idx] += aux[featId][idx];
                }
            }
        }
    }

    inline void loss(const int& featId, const Tensor1D& loss) {
        if (loss.dim != val.row) {
            std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
        }
        indexers[featId] = true;
        for (int idx = 0; idx < val.row; idx++) {
            grad[featId][idx] += loss[idx];
        }
    }

    inline void loss(const vector<int>& featIds, const Tensor1D& loss) {
        if (loss.dim != val.row) {
            std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
        }
        int featNum = featIds.size();
        int featId;
        for (int i = 0; i < featNum; i++) {
            featId = featIds[i];
            indexers[featId] = true;
            for (int idx = 0; idx < val.row; idx++) {
                grad[featId][idx] += loss[idx];
            }
        }
    }

    inline void save(std::ofstream &os)const {
        val.save(os);
        aux.save(os);
        os << max_update << std::endl;
        os << val.col << std::endl;
        os << last_update[0];
        for (int idx = 1; idx < val.col; idx++) {
            os << " " << last_update[idx];
        }
        os << std::endl;
    }


    inline void load(std::ifstream &is) {
        val.load(is);
        aux.load(is);
        is >> max_update;
        int curInDim;
        is >> curInDim;
        last_update.resize(curInDim);
        for (int idx = 0; idx < curInDim; idx++) {
            is >> last_update[idx];
        }
    }

};

#endif /* AVGPARAM_H_ */
