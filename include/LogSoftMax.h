#ifndef LOG_SOFT_MAX_DRIVER
#define LOG_SOFT_MAX_DRIVER

/*
*  LOG_SOFT_MAX_DRIVER.h:
*  a contextual builder, concatenate x[-c]...x[0]...x[c] together
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "PAddOP.h"
#include "Graph.h"

class LogSoftMax {
  public:
    int _nSize;

    vector<PSubNode> _middles;
    vector<ActivateNode> _expmiddles;
    vector<PSubNode> _outputs;
    PAddNode _sum;
    ActivateNode _logsum;

  public:
    LogSoftMax() {
        clear();
    }

    ~LogSoftMax() {
        clear();
    }


    void clear() {
        _outputs.clear();
        _middles.clear();
        _expmiddles.clear();
        _nSize = 0;
    }


    void init(int maxsize) {
        _middles.resize(maxsize);
        _expmiddles.resize(maxsize);
        _outputs.resize(maxsize);
        for (int idx = 0; idx < maxsize; idx++) {
            _middles[idx].init(1, -1);
            _outputs[idx].init(1, -1);
            _expmiddles[idx].init(1, -1);
            _expmiddles[idx].setFunctions(&fexp, &dexp);
        }
        _sum.init(1, -1);
        _logsum.init(1, -1);
        _logsum.setFunctions(&flog, &dlog);
    }



  public:
    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for LOG_SOFT_MAX_DRIVER operation" << std::endl;
            return;
        }

        _nSize = x.size();
        for (int idx = 0; idx < _nSize; idx++) {
            if (x[idx]->dim != 1) {
                std::cout << "the dim of input nodes for LOG_SOFT_MAX_DRIVER is not 1" << std::endl;
                return;
            }
        }

        PNode pmax_node = x[0];
        for (int idx = 1; idx < _nSize; idx++) {
            if (x[idx]->val[0] > pmax_node->val[0]) {
                pmax_node = x[idx];
            }
        }

        for (int idx = 0; idx < _nSize; idx++) {
            _middles[idx].forward(cg, x[idx], pmax_node);
            _expmiddles[idx].forward(cg, &_middles[idx]);
        }

        _sum.forward(cg, toPointers<ActivateNode, Node>(_expmiddles, _nSize));
        _logsum.forward(cg, &_sum);

        for (int idx = 0; idx < _nSize; idx++) {
            _outputs[idx].forward(cg, &_middles[idx], &_logsum);
        }
    }

};


#endif
