#ifndef _LOOKUPTABLE_H_
#define _LOOKUPTABLE_H_

/*
 *  LookupTable.h:
 *  Lookup operation, for embeddings
 *
 *  Created on: Apr 22, 2017
 *      Author: mszhang
 */

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include "profiler.h"

class LookupTable
#if USE_GPU
: public TransferableComponents
#endif
{
public:
    PAlphabet elems;
    SparseParam E;
    bool bFineTune;
    int nDim;
    int nVSize;
    int nUNKId;

    LookupTable() {
        nVSize = 0;
        nDim = 0;
        elems = NULL;
        nUNKId = -1;
        bFineTune = false;
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() {
        return {&E};
    }

    virtual std::string name() const {
        return "LookupTable";
    }
#endif

    //random initialization
    void initial(PAlphabet alpha, int dim, bool fineTune = true) {
        elems = alpha;
        nVSize = elems->size();
        nUNKId = elems->from_string(unknownkey);
        initialWeights(dim, fineTune);
    }

    //initialization by pre-trained embeddings
    bool initial(PAlphabet alpha, const string& inFile, bool fineTune = true, dtype norm = -1) {
        elems = alpha;
        nVSize = elems->size();
        nUNKId = elems->from_string(unknownkey);
        return initialWeights(inFile, fineTune, norm);
    }

    void initialWeights(int dim, bool tune) {
        if (dim <=0 || nVSize == 0 || (nVSize == 1 && nUNKId >= 0)) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        nDim = dim;
        E.initial(nDim, nVSize);
        E.val.random(sqrt(1.0 / nDim));
        //E.val.norm2one();
        bFineTune = tune;
#if USE_GPU
        E.val.copyFromHostToDevice();
#endif
    }

    // default should be fineTune, just for initialization
    bool initialWeights(const string& inFile, bool tune, dtype norm = -1) {
        if (nVSize == 0 || !elems->is_fixed() || (nVSize == 1 && nUNKId >= 0)) {
            std::cout << "please check the alphabet" << std::endl;
            return false;
        }

        ifstream inf;
        if (inf.is_open()) {
            inf.close();
            inf.clear();
        }
        inf.open(inFile.c_str());

        if (!inf.is_open()) {
            std::cout << "please check the input file" << std::endl;
            return false;
        }

        string strLine, curWord;
        int wordId;

        vector<string> sLines;
        sLines.clear();
        while (1) {
            if (!my_getline(inf, strLine)) {
                break;
            }
            if (!strLine.empty()) {
                sLines.push_back(strLine);
            }
        }
        inf.close();
        if (sLines.size() == 0) {
            return false;
        }

        //find the first line, decide the wordDim;
		vector<string>::iterator it = sLines.begin();
		sLines.erase(it);
        vector<string> vecInfo;
        split_bychar(sLines[0], vecInfo, ' ');
        nDim = vecInfo.size() - 1;

        E.initial(nDim, nVSize);

        std::cout << "word embedding dim is " << nDim << std::endl;

        bool bHasUnknown = false;
        unordered_set<int> indexers;
        NRVec<dtype> sum(nDim);
        sum = 0.0;
        int count = 0;
        for (int idx = 0; idx < sLines.size(); idx++) {
            split_bychar(sLines[idx], vecInfo, ' ');
            if (vecInfo.size() != nDim + 1) {
                std::cout << "error embedding file" << std::endl;
            }
            curWord = vecInfo[0];
            //we assume the keys are normalized
            wordId = elems->from_string(curWord);
            if (wordId >= 0) {
                count++;
                if (nUNKId == wordId) {
                    bHasUnknown = true;
                }
                indexers.insert(wordId);

                for (int idy = 0; idy < nDim; idy++) {
                    dtype curValue = atof(vecInfo[idy + 1].c_str());
                    sum[idy] += curValue;
                    E.val[wordId][idy] += curValue;
                }
            }
        }

        if (count == 0) {
            E.val.random(sqrt(3.0 / nDim));
            std::cout << "find no overlapped lexicons in the embedding file" << std::endl;
            return false;
        }

        if (nUNKId >= 0 && !bHasUnknown) {
            for (int idx = 0; idx < nDim; idx++) {
                E.val[nUNKId][idx] = sum[idx] / (count + 1);
            }
            indexers.insert(nUNKId);
            count++;
            std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
        }


        int oovWords = 0;
        for (int id = 0; id < nVSize; id++) {
            if (indexers.find(id) == indexers.end()) {
                oovWords++;
                for (int idy = 0; idy < nDim; idy++) {
                    E.val[id][idy] = nUNKId >= 0 ? E.val[nUNKId][idy] : sum[idy] / (count + 1);
                }
            }
        }


        std::cout << "OOV num is " << oovWords << ", total num is " << nVSize << ", embedding oov ratio is " << oovWords * 1.0 / nVSize << std::endl;
        std::cout << "unknown id" << nUNKId << std::endl;
        bFineTune = tune;
        if (norm > 0) {
            E.val.norm2one(norm);
        }
#if USE_GPU
        E.val.copyFromHostToDevice();
#endif
        return true;
    }

    void exportAdaParams(ModelUpdate& ada) {
        if (bFineTune) {
            ada.addParam(&E);
        }
    }


    int getElemId(const string& strFeat) const {
        return elems->from_string(strFeat);
    }

    void save(std::ofstream &os) const {
        E.save(os);
        os << bFineTune << std::endl;
        os << nDim << std::endl;
        os << nVSize << std::endl;
        os << nUNKId << std::endl;
    }

    //set alpha directly
    void load(std::ifstream &is, PAlphabet alpha) {
        E.load(is);
        is >> bFineTune;
        is >> nDim;
        is >> nVSize;
        is >> nUNKId;
        elems = alpha;
    }

    void load(std::ifstream &is, Alphabet &alpha) {
        this->load(is, &alpha);
    }
};


class LookupNode : public Node {
public:
    LookupTable* param;
    int xid;

    LookupNode() {
        xid = -1;
        param = NULL;
        node_type = "lookup";
    }

    void setParam(LookupTable* paramInit) {
        param = paramInit;
    }

    void setParam(LookupTable &table) {
        param = &table;
    }

    //notice the output
    //this should be leaf nodes
    void forward(Graph *cg, const string& strNorm) {
        assert(param != NULL);
        xid = param->getElemId(strNorm);
        if (xid < 0 && param->nUNKId >= 0) {
            xid = param->nUNKId;
        }
        if (param->bFineTune && xid < 0) {
            std::cout << "Caution: unknown words are not modeled !" << std::endl;
        }
        degree = 0;
        cg->addNode(this);
    }

    void forward(Graph &graph, const string &word) {
        this->forward(&graph, word);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LookupNode* conv_other = (LookupNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode() ^ ::typeHashCode(param);
    }

    // for which do no require merge
    void compute() {
        if (xid >= 0) {
            param->E.value(xid, val);
        } else {
            val.zero();
        }
    }

    void backward() {
        assert(param != NULL);
        if (xid == param->nUNKId || (xid >= 0 && param->bFineTune)) {
            param->E.loss(xid, loss);
        }
    }
};


#if USE_GPU
class LookupExecute :public Execute {
public:
    int dim;
    Tensor2D drop_mask;
    LookupTable *table;
    std::vector<int> xids;

    void  forward() {
        int count = batch.size();
        drop_mask.init(dim, count);
        CalculateDropMask(count, dim, drop_mask);
        xids.reserve(count);
        std::vector<dtype*> vals;
        vals.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            LookupNode *n = static_cast<LookupNode*>(batch[idx]);
            xids.push_back(n->xid);
            vals.push_back(n->val.value);
        }

        n3ldg_cuda::LookupForward(xids, table->E.val.value, bTrain,
                drop_mask.value, dynamicDropValue(), count, dim, vals);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[i][j];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ?
                    0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
            int xid = static_cast<LookupNode*>(batch[idx])->xid;
            n3ldg_cuda::Assert(batch[idx]->val.verify("lookup forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (Node *n : batch) {
            losses.push_back(n->loss.value);
        }
        n3ldg_cuda::LookupBackward(xids, table->nUNKId, table->bFineTune,
                losses,
                drop_mask.value,
                dynamicDropValue(),
                count,
                dim,
                table->E.grad.value,
                table->E.dIndexers.value);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }

        n3ldg_cuda::Assert(table->E.grad.verify("lookup backward grad"));
        n3ldg_cuda::Assert(n3ldg_cuda::Verify(table->E.indexers.c_buf(),
                    table->E.dIndexers.value,
                    table->E.dIndexers.len,
                    "lookup backward index"));
#endif
    }
};
#else
class LookupExecute :public Execute {
    public:
        void  forward() {
            int count = batch.size();
            //#pragma omp parallel for
            for (int idx = 0; idx < count; idx++) {
                batch[idx]->compute();
                batch[idx]->forward_drop(bTrain, drop_factor);
            }
        }

        void backward() {
            int count = batch.size();
            //#pragma omp parallel for
            for (int idx = 0; idx < count; idx++) {
                batch[idx]->backward_drop();
                batch[idx]->backward();
            }
        }
};
#endif


PExecute LookupNode::generate(bool bTrain, dtype cur_drop_factor) {
    LookupExecute* exec = new LookupExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
#if USE_GPU
    exec->table = param;
    exec->dim = dim;
#endif
    return exec;
}

#endif /*_LOOKUPTABLE_H*/
