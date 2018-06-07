#ifndef LSTM1
#define LSTM1

/*
*  LSTM1.h:
*  LSTM variation 1
*
*  Created on: June 13, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"

struct LSTM1Params {
    BiParams input;
    BiParams output;
    BiParams forget;
    BiParams cell;

    LSTM1Params() {
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        input.exportAdaParams(ada);
        output.exportAdaParams(ada);
        forget.exportAdaParams(ada);
        cell.exportAdaParams(ada);
    }

    inline void initial(int nOSize, int nISize) {
        input.initial(nOSize, nOSize, nISize, true);
        output.initial(nOSize, nOSize, nISize, true);
        forget.initial(nOSize, nOSize, nISize, true);
        cell.initial(nOSize, nOSize, nISize, true);

    }

    inline int inDim() {
        return input.W2.inDim();
    }

    inline int outDim() {
        return input.W2.outDim();
    }

    inline void save(std::ofstream &os) const {
        input.save(os);
        output.save(os);
        forget.save(os);
        cell.save(os);
    }

    inline void load(std::ifstream &is) {
        input.load(is);
        output.load(is);
        forget.load(is);
        cell.load(is);
    }

};

// standard LSTM1 using tanh as activation function
// other conditions are not implemented unless they are clear
class LSTM1Builder {
  public:
    int _nSize;
    int _inDim;
    int _outDim;

    vector<BiNode> _inputgates;
    vector<BiNode> _forgetgates;
    vector<BiNode> _halfcells;

    vector<PMultiNode> _inputfilters;
    vector<PMultiNode> _forgetfilters;

    vector<PAddNode> _cells;
    vector<BiNode> _outputgates;
    vector<TanhNode> _halfhiddens;
    vector<PMultiNode> _hiddens;  // intermediate result without dropout

    BucketNode _bucket;

    LSTM1Params* _param;

    bool _left2right;

  public:
    LSTM1Builder() {
        clear();
    }

    ~LSTM1Builder() {
        clear();
    }

  public:
    inline void init(LSTM1Params* paramInit, dtype dropout, bool left2right = true) {
        _param = paramInit;
        _inDim = _param->input.W2.inDim();
        _outDim = _param->input.W2.outDim();
        int maxsize = _inputgates.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates[idx].setParam(&_param->input);
            _forgetgates[idx].setParam(&_param->forget);
            _outputgates[idx].setParam(&_param->output);
            _halfcells[idx].setParam(&_param->cell);
            _inputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
            _forgetgates[idx].setFunctions(&fsigmoid, &dsigmoid);
            _outputgates[idx].setFunctions(&fsigmoid, &dsigmoid);
            _halfcells[idx].setFunctions(&ftanh, &dtanh);
        }
        _left2right = left2right;

        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates[idx].init(_outDim, -1);
            _forgetgates[idx].init(_outDim, -1);
            _halfcells[idx].init(_outDim, -1);
            _inputfilters[idx].init(_outDim, -1);
            _forgetfilters[idx].init(_outDim, -1);
            _cells[idx].init(_outDim, -1);
            _outputgates[idx].init(_outDim, -1);
            _halfhiddens[idx].init(_outDim, -1);
            _hiddens[idx].init(_outDim, dropout);
        }

        _bucket.init(_outDim, -1);

    }

    inline void resize(int maxsize) {
        _inputgates.resize(maxsize);
        _forgetgates.resize(maxsize);
        _halfcells.resize(maxsize);
        _inputfilters.resize(maxsize);
        _forgetfilters.resize(maxsize);
        _cells.resize(maxsize);
        _outputgates.resize(maxsize);
        _halfhiddens.resize(maxsize);
        _hiddens.resize(maxsize);
    }

    //whether vectors have been allocated
    inline bool empty() {
        return _hiddens.empty();
    }

    inline void clear() {
        _inputgates.clear();
        _forgetgates.clear();
        _halfcells.clear();
        _inputfilters.clear();
        _forgetfilters.clear();
        _cells.clear();
        _outputgates.clear();
        _halfhiddens.clear();
        _hiddens.clear();

        _left2right = true;
        _param = NULL;
        _nSize = 0;
        _inDim = 0;
        _outDim = 0;
    }

  public:
    inline void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x[0]->val.dim != _inDim) {
            std::cout << "input dim does not match for lstm operation" << std::endl;
            return;
        }

        if (_left2right) {
            left2right_forward(cg, x);
        } else {
            right2left_forward(cg, x);
        }
    }

  protected:
    inline void left2right_forward(Graph *cg, const vector<PNode>& x) {
        for (int idx = 0; idx < _nSize; idx++) {
            if (idx == 0) {
                _bucket.forward(cg, 0);

                _inputgates[idx].forward(cg, &_bucket, x[idx]);

                _halfcells[idx].forward(cg, &_bucket, x[idx]);

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

                _cells[idx].forward(cg, &_inputfilters[idx], &_bucket);

                _halfhiddens[idx].forward(cg, &_cells[idx]);

                _outputgates[idx].forward(cg, &_bucket, x[idx]);

                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);

            } else {
                _inputgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

                _forgetgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

                _halfcells[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

                _forgetfilters[idx].forward(cg, &_cells[idx - 1], &_forgetgates[idx]);

                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

                _halfhiddens[idx].forward(cg, &_cells[idx]);

                _outputgates[idx].forward(cg, &_hiddens[idx - 1], x[idx]);

                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            }
        }
    }

    inline void right2left_forward(Graph *cg, const vector<PNode>& x) {
        for (int idx = _nSize - 1; idx >= 0; idx--) {
            if (idx == _nSize - 1) {
                _bucket.forward(cg, 0);

                _inputgates[idx].forward(cg, &_bucket, x[idx]);

                _halfcells[idx].forward(cg, &_bucket, x[idx]);

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

                _cells[idx].forward(cg, &_inputfilters[idx], &_bucket);

                _halfhiddens[idx].forward(cg, &_cells[idx]);

                _outputgates[idx].forward(cg, &_bucket, x[idx]);

                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            } else {
                _inputgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

                _forgetgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

                _halfcells[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);

                _forgetfilters[idx].forward(cg, &_cells[idx + 1], &_forgetgates[idx]);

                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);

                _halfhiddens[idx].forward(cg, &_cells[idx]);

                _outputgates[idx].forward(cg, &_hiddens[idx + 1], x[idx]);

                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            }

        }
    }
};


class IncLSTM1Builder {
  public:
    int _nSize;
    int _inDim;
    int _outDim;

    IncLSTM1Builder* _pPrev;

    BiNode _inputgate;
    BiNode _forgetgate;
    BiNode _halfcell;

    PMultiNode _inputfilter;
    PMultiNode _forgetfilter;

    PAddNode _cell;

    BiNode _outputgate;

    TanhNode _halfhidden;

    PMultiNode _hidden;  // intermediate result without dropout

    BucketNode _bucket;

    LSTM1Params* _param;

  public:
    IncLSTM1Builder() {
        clear();
    }

    ~IncLSTM1Builder() {
        clear();
    }

    void clear() {
        _nSize = 0;
        _inDim = 0;
        _outDim = 0;
        _param = NULL;
        _pPrev = NULL;
    }

  public:
    inline void init(LSTM1Params* paramInit, dtype dropout) {
        _param = paramInit;
        _inDim = _param->input.W2.inDim();
        _outDim = _param->input.W2.outDim();

        _inputgate.setParam(&_param->input);
        _forgetgate.setParam(&_param->forget);
        _outputgate.setParam(&_param->output);
        _halfcell.setParam(&_param->cell);
        _inputgate.setFunctions(&fsigmoid, &dsigmoid);
        _forgetgate.setFunctions(&fsigmoid, &dsigmoid);
        _outputgate.setFunctions(&fsigmoid, &dsigmoid);
        _halfcell.setFunctions(&ftanh, &dtanh);

        _inputgate.init(_outDim, -1);
        _forgetgate.init(_outDim, -1);
        _halfcell.init(_outDim, -1);
        _inputfilter.init(_outDim, -1);
        _forgetfilter.init(_outDim, -1);
        _cell.init(_outDim, -1);
        _outputgate.init(_outDim, -1);
        _halfhidden.init(_outDim, -1);
        _hidden.init(_outDim, dropout);

        _bucket.init(_outDim, -1);
    }


  public:
    inline void forward(Graph *cg, PNode x, IncLSTM1Builder* prev = NULL) {
        if (prev == NULL) {
            _bucket.forward(cg, 0);

            _inputgate.forward(cg, &_bucket, x);

            _halfcell.forward(cg, &_bucket, x);

            _inputfilter.forward(cg, &_halfcell, &_inputgate);

            _cell.forward(cg, &_inputfilter, &_bucket);

            _halfhidden.forward(cg, &_cell);

            _outputgate.forward(cg, &_bucket, x);

            _hidden.forward(cg, &_halfhidden, &_outputgate);

            _nSize = 1;
        } else {
            _inputgate.forward(cg, &(prev->_hidden), x);

            _forgetgate.forward(cg, &(prev->_hidden), x);

            _halfcell.forward(cg, &(prev->_hidden), x);

            _inputfilter.forward(cg, &_halfcell, &_inputgate);

            _forgetfilter.forward(cg, &(prev->_cell), &_forgetgate);

            _cell.forward(cg, &_inputfilter, &_forgetfilter);

            _halfhidden.forward(cg, &_cell);

            _outputgate.forward(cg, &(prev->_hidden), x);

            _hidden.forward(cg, &_halfhidden, &_outputgate);

            _nSize = prev->_nSize + 1;
        }

        _pPrev = prev;
    }

};


#endif
