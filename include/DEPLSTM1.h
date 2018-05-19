#ifndef DEPLSTM1
#define DEPLSTM1

#include "MyLib.h"
#include "Node.h"
#include "TriOP.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"

struct TreeLSTM1Params {
    UniParams input_l;
    UniParams input_r;
    UniParams output_l;
    UniParams output_r;
    UniParams forget_l;
    UniParams forget_r;
    UniParams cell_l;
    UniParams cell_r;

    TreeLSTM1Params() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        input_l.exportAdaParams(ada);
        output_l.exportAdaParams(ada);
        forget_l.exportAdaParams(ada);
        cell_l.exportAdaParams(ada);
        input_r.exportAdaParams(ada);
        output_r.exportAdaParams(ada);
        forget_r.exportAdaParams(ada);
        cell_r.exportAdaParams(ada);
    }

    void initial(int nOSize, int nISize, PAlphabet label) {
        input_l.initial(nOSize, nISize, true);
        output_l.initial(nOSize, nISize, true);
        forget_l.initial(nOSize, nISize, true);
        cell_l.initial(nOSize, nISize, true);

        input_r.initial(nOSize, nOSize);
        output_r.initial(nOSize, nOSize);
        forget_r.initial(nOSize, nOSize);
        cell_r.initial(nOSize, nOSize);
    }

    int inDim() {
        return input_l.W.inDim();
    }

    int outDim() {
        return input_l.W.outDim();
    }

    void save(std::ofstream &os) const {

    }

    void load(std::ifstream &is) {

    }

};

// standard TreeLSTM1Builder from bottom to top
class TreeLSTM1Builder {
  public:
    int _nSize;
    int _inDim;
    int _outDim;

    bool _bottom2top;

    vector<LinearNode> _inputgates_l;
    vector<LinearNode> _forgetgates_l;
    vector<LinearNode> _halfcells_l;
    vector<LinearNode> _outputgates_l;

    vector<UniNode> _inputgates_r;
    vector<UniNode> _forgetgates_r;
    vector<UniNode> _halfcells_r;
    vector<UniNode> _outputgates_r;

    vector<PAddNode> _inputgates_add;
    vector<PAddNode> _forgetgates_add;
    vector<PAddNode> _halfcells_add;
    vector<PAddNode> _outputgates_add;

    vector<SigmoidNode> _inputgates;
    vector<SigmoidNode> _forgetgates;
    vector<TanhNode> _halfcells;
    vector<SigmoidNode> _outputgates;

    vector<PMultiNode> _inputfilters;
    vector<PMultiNode> _forgetfilters;

    vector<PAddNode> _cells;

    vector<TanhNode> _halfhiddens;

    vector<PMultiNode> _hiddens;

    TreeLSTM1Params* _param;


  public:
    TreeLSTM1Builder() {
        clear();
    }

    ~TreeLSTM1Builder() {
        clear();
    }

  public:
    void init(TreeLSTM1Params* paramInit, dtype dropout, bool bottom2top = true) {
        _param = paramInit;
        _inDim = _param->input_l.W.inDim();
        _outDim = _param->input_l.W.outDim();
        int maxsize = _inputgates.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates_l[idx].setParam(&(_param->input_l));
            _forgetgates_l[idx].setParam(&(_param->forget_l));
            _outputgates_l[idx].setParam(&(_param->output_l));
            _halfcells_l[idx].setParam(&(_param->cell_l));

            _inputgates_r[idx].setParam(&(_param->input_r));
            _forgetgates_r[idx].setParam(&(_param->forget_r));
            _outputgates_r[idx].setParam(&(_param->output_r));
            _halfcells_r[idx].setParam(&(_param->cell_r));

        }

        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates_l[idx].init(_outDim, -1);
            _forgetgates_l[idx].init(_outDim, -1);
            _outputgates_l[idx].init(_outDim, -1);
            _halfcells_l[idx].init(_outDim, -1);

            _inputgates_r[idx].init(_outDim, -1);
            _forgetgates_r[idx].init(_outDim, -1);
            _outputgates_r[idx].init(_outDim, -1);
            _halfcells_r[idx].init(_outDim, -1);

            _inputgates_add[idx].init(_outDim, -1);
            _forgetgates_add[idx].init(_outDim, -1);
            _outputgates_add[idx].init(_outDim, -1);
            _halfcells_add[idx].init(_outDim, -1);

            _inputgates[idx].init(_outDim, -1);
            _forgetgates[idx].init(_outDim, -1);
            _outputgates[idx].init(_outDim, -1);
            _halfcells[idx].init(_outDim, -1);

            _inputfilters[idx].init(_outDim, -1);
            _forgetfilters[idx].init(_outDim, -1);
            _cells[idx].init(_outDim, -1);
            _halfhiddens[idx].init(_outDim, -1);
            _hiddens[idx].init(_outDim, dropout);
        }

        _bottom2top = bottom2top;
    }

    void resize(int maxsize) {
        _inputgates_l.resize(maxsize);
        _forgetgates_l.resize(maxsize);
        _halfcells_l.resize(maxsize);
        _outputgates_l.resize(maxsize);

        _inputgates_r.resize(maxsize);
        _forgetgates_r.resize(maxsize);
        _halfcells_r.resize(maxsize);
        _outputgates_r.resize(maxsize);

        _inputgates_add.resize(maxsize);
        _forgetgates_add.resize(maxsize);
        _halfcells_add.resize(maxsize);
        _outputgates_add.resize(maxsize);

        _inputgates.resize(maxsize);
        _forgetgates.resize(maxsize);
        _halfcells.resize(maxsize);
        _outputgates.resize(maxsize);

        _inputfilters.resize(maxsize);
        _forgetfilters.resize(maxsize);
        _cells.resize(maxsize);
        _halfhiddens.resize(maxsize);
        _hiddens.resize(maxsize);
    }


    void clear() {
        _inputgates_l.clear();
        _forgetgates_l.clear();
        _halfcells_l.clear();
        _outputgates_l.clear();

        _inputgates_r.clear();
        _forgetgates_r.clear();
        _halfcells_r.clear();
        _outputgates_r.clear();

        _inputgates_add.clear();
        _forgetgates_add.clear();
        _halfcells_add.clear();
        _outputgates_add.clear();

        _inputgates.clear();
        _forgetgates.clear();
        _halfcells.clear();
        _outputgates.clear();

        _inputfilters.clear();
        _forgetfilters.clear();
        _cells.clear();
        _halfhiddens.clear();
        _hiddens.clear();

        _param = NULL;
        _nSize = 0;
        _inDim = 0;
        _outDim = 0;
        _bottom2top = true;
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x, const vector<int>& heads) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }

        _nSize = x.size();

        if (_bottom2top) {
            btforward(cg, x, heads);
        } else {
            tbforward(cg, x, heads);
        }
    }

  protected:
    void btforward(Graph *cg, const vector<PNode>& x, const vector<int>& heads) {
        vector<vector<int> > children;
        vector<bool> computed;
        children.resize(_nSize);
        computed.resize(_nSize);

        for (int idx = 0; idx < _nSize; idx++) {
            children[idx].clear();
            computed[idx] = false;
        }

        for (int idx = 0; idx < _nSize; idx++) {
            int curHead = heads[idx];
            if (curHead >= 0) {
                children[curHead].push_back(idx);
            }
        }

        vector<PNode> sumInputNodes, sumOutputNodes, sumHaffCellNodes;
        vector<PNode> updatedCellNodes;

        int computed_count = 0;
        while (computed_count < _nSize) {
            for (int idx = 0; idx < _nSize; idx++) {
                if (computed[idx])continue;
                int child_num = children[idx].size();
                bool allcomputed = true;
                for (int idy = 0; idy < child_num; idy++) {
                    if (!computed[children[idx][idy]]) {
                        allcomputed = false;
                        break;
                    }
                }
                if (!allcomputed)continue;

                sumInputNodes.clear();
                sumOutputNodes.clear();
                sumHaffCellNodes.clear();
                updatedCellNodes.clear();
                for (int idy = 0; idy < child_num; idy++) {
                    int curChild = children[idx][idy];
                    _inputgates_r[curChild].forward(cg, &_hiddens[curChild]);
                    sumInputNodes.push_back(&(_inputgates_r[curChild]));

                    _outputgates_r[curChild].forward(cg, &_hiddens[curChild]);
                    sumOutputNodes.push_back(&(_outputgates_r[curChild]));

                    _halfcells_r[curChild].forward(cg, &_hiddens[curChild]);
                    sumHaffCellNodes.push_back(&(_halfcells_r[curChild]));
                }

                _inputgates_l[idx].forward(cg, x[idx]);
                sumInputNodes.push_back(&(_inputgates_l[idx]));
                _inputgates_add[idx].forward(cg, sumInputNodes);
                _inputgates[idx].forward(cg, &(_inputgates_add[idx]));

                _outputgates_l[idx].forward(cg, x[idx]);
                sumOutputNodes.push_back(&(_outputgates_l[idx]));
                _outputgates_add[idx].forward(cg, sumOutputNodes);
                _outputgates[idx].forward(cg, &(_outputgates_add[idx]));

                _halfcells_l[idx].forward(cg, x[idx]);
                sumHaffCellNodes.push_back(&(_halfcells_l[idx]));
                _halfcells_add[idx].forward(cg, sumHaffCellNodes);
                _halfcells[idx].forward(cg, &(_halfcells_add[idx]));

                _forgetgates_l[idx].forward(cg, x[idx]);
                vector<PNode> updatedCellNodes;
                for (int idy = 0; idy < child_num; idy++) {
                    int curChild = children[idx][idy];
                    _forgetgates_r[curChild].forward(cg, &_hiddens[curChild]);
                    _forgetgates_add[curChild].forward(cg, &(_forgetgates_l[idx]), &(_forgetgates_r[curChild]));
                    _forgetgates[curChild].forward(cg, &(_forgetgates_add[curChild]));
                    _forgetfilters[curChild].forward(cg, &(_forgetgates[curChild]), &(_cells[curChild]));
                    updatedCellNodes.push_back(&(_forgetfilters[curChild]));
                }

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                updatedCellNodes.push_back(&(_inputfilters[idx]));

                _cells[idx].forward(cg, updatedCellNodes);

                _halfhiddens[idx].forward(cg, &_cells[idx]);

                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);

                computed_count++;
                computed[idx] = true;
            }
        }
    }

    void tbforward(Graph *cg, const vector<PNode>& x, const vector<int>& heads) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }

        _nSize = x.size();
        vector<vector<int> > children;
        vector<bool> computed;
        children.resize(_nSize);
        computed.resize(_nSize);

        for (int idx = 0; idx < _nSize; idx++) {
            children[idx].clear();
            computed[idx] = false;
        }

        int root = -1;
        for (int idx = 0; idx < _nSize; idx++) {
            int curHead = heads[idx];
            if (curHead >= 0) {
                children[curHead].push_back(idx);
            } else {
                root = idx;
            }
        }

        vector<PNode> sumInputNodes, sumOutputNodes, sumForgetNodes, sumHaffCellNodes;
        vector<PNode> updatedCellNodes;

        vector<int> queue;
        queue.push_back(root);
        int position = 0;
        while (position < queue.size()) {
            int curId = queue[position];
            int curHead = heads[curId];

            sumInputNodes.clear();
            sumOutputNodes.clear();
            sumForgetNodes.clear();
            sumHaffCellNodes.clear();

            _inputgates_l[curId].forward(cg, x[curId]);
            sumInputNodes.push_back(&(_inputgates_l[curId]));

            _outputgates_l[curId].forward(cg, x[curId]);
            sumOutputNodes.push_back(&(_outputgates_l[curId]));
            _halfcells_l[curId].forward(cg, x[curId]);
            sumHaffCellNodes.push_back(&(_halfcells_l[curId]));

            if (curHead >= 0) {
                _inputgates_r[curId].forward(cg, &(_hiddens[curHead]));
                sumInputNodes.push_back(&(_inputgates_r[curId]));

                _outputgates_r[curId].forward(cg, &(_hiddens[curHead]));
                sumOutputNodes.push_back(&(_outputgates_r[curId]));
                _halfcells_r[curId].forward(cg, &(_hiddens[curHead]));
                sumHaffCellNodes.push_back(&(_halfcells_r[curId]));

                _forgetgates_l[curId].forward(cg, x[curId]);
                sumForgetNodes.push_back(&(_forgetgates_l[curId]));
                _forgetgates_r[curId].forward(cg, &(_hiddens[curHead]));
                sumForgetNodes.push_back(&(_forgetgates_r[curId]));

                _forgetgates_add[curId].forward(cg, sumForgetNodes);
                _forgetgates[curId].forward(cg, &(_forgetgates_add[curId]));

                _forgetfilters[curId].forward(cg, &(_forgetgates[curId]), &(_cells[curHead]));

                updatedCellNodes.push_back(&(_forgetfilters[curId]));
            }

            _inputgates_add[curId].forward(cg, sumInputNodes);
            _inputgates[curId].forward(cg, &(_inputgates_add[curId]));

            _halfcells_add[curId].forward(cg, sumHaffCellNodes);
            _halfcells[curId].forward(cg, &(_halfcells_add[curId]));

            _inputfilters[curId].forward(cg, &(_inputgates[curId]), &(_halfcells[curId]));

            updatedCellNodes.push_back(&(_inputfilters[curId]));

            _cells[curId].forward(cg, updatedCellNodes);

            _outputgates_add[curId].forward(cg, sumOutputNodes);
            _outputgates[curId].forward(cg, &(_outputgates_add[curId]));

            _halfhiddens[curId].forward(cg, &_cells[curId]);

            _hiddens[curId].forward(cg, &_halfhiddens[curId], &_outputgates[curId]);

            for (int idx = 0; idx < children[curId].size(); idx++) {
                queue.push_back(children[curId][idx]);
            }
            position++;
        }
    }


};

#endif
