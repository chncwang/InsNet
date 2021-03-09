#ifndef _LOOKUPTABLE_H_
#define _LOOKUPTABLE_H_

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include "profiler.h"
#include "boost/format.hpp"
#include "Param.h"

using boost::format;

template<typename ParamType>
class LookupTable : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    Alphabet elems;
    ParamType E;
    bool bFineTune;
    int nDim;
    int nVSize;
    int nUNKId;
    bool inited = false;

    LookupTable(const string &name = "embedding") : E(name) {
        nVSize = 0;
        nDim = 0;
        nUNKId = -1;
        bFineTune = false;
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&E};
    }

    virtual std::string name() const {
        return "LookupTable";
    }
#endif

    void init(const Alphabet &alpha, int dim, bool fineTune = true) {
        if (!inited) {
            elems = alpha;
            nVSize = elems.size();
            nUNKId = elems.from_string(unknownkey);
            initWeights(dim, fineTune);
            inited = true;
        }
    }

    //initialization by pre-trained embeddings
    void init(const Alphabet &alpha, const string& inFile, bool fineTune = true, dtype norm = -1) {
        elems = alpha;
        nVSize = elems.size();
        nUNKId = elems.from_string(unknownkey);
        initWeights(inFile, fineTune, norm);
    }

    void initWeights(int dim, bool tune) {
        if (dim <=0 || nVSize == 0 || (nVSize == 1 && nUNKId >= 0)) {
            std::cerr << boost::format("LookupTable initWeights - dim:%1% size:%2%") % dim % nVSize
                << endl;
            std::cerr << "please check the alphabet" << std::endl;
            abort();
        }
        nDim = dim;
        cout << format("initWeights dim:%1% vocabulary_size:%2%\n") % nDim % nVSize;
        E.init(nDim, nVSize);
        E.val.random(sqrt(1.0 / nDim));
        //E.val.norm2one();
        bFineTune = tune;
#if USE_GPU
        E.val.copyFromHostToDevice();
#endif
    }

    // default should be fineTune, just for initialization
    void initWeights(const string& inFile, bool tune, dtype norm = -1) {
        if (nVSize == 0 || (nVSize == 1 && nUNKId >= 0)) {
            cout << "nVSize:" << nVSize << " nUNKId:" << nUNKId << endl;
            std::cerr << "please check the alphabet" << std::endl;
            abort();
        }

        ifstream inf;
        inf.open(inFile.c_str());

        if (!inf.is_open()) {
            std::cerr << "please check the input file" << std::endl;
            abort();
        }

        string strLine, curWord;
        vector<string> sLines;
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
            cerr << "sLines size is 0" << endl;
            abort();
        }

        //find the first line, decide the wordDim;
		vector<string>::iterator it = sLines.begin();
		sLines.erase(it);
        vector<string> vecInfo;
        split_bychar(sLines[0], vecInfo, ' ');
        nDim = vecInfo.size() - 1;

        cout << format("nDim:%1% nVSize:%2%") % nDim % nVSize << endl;
        E.init(nDim, nVSize);

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
            if (elems.find_string(curWord)) {
                int wordId = elems.insert_string(curWord);
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
            abort();
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
    }

    std::vector<Tunable<BaseParam>*> tunableComponents() override {
        if (bFineTune) {
            return {&E};
        } else {
            return {};
        }
    }

    int getElemId(const string& strFeat) const {
        return elems.find_string(strFeat) ? elems.from_string(strFeat) : nUNKId;
    }

    bool findElemId(const string &str) const {
        return elems.find_string(str);
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["e"] = E.toJson();
        json["finetune"] = bFineTune;
        json["dim"] = nDim;
        json["vocabulary_size"] = nVSize;
        json["unkown_id"] = nUNKId;
        json["word_ids"] = elems.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        bFineTune = json["finetune"].asBool();
        nDim = json["dim"].asInt();
        nVSize = json["vocabulary_size"].asInt();
        nUNKId = json["unkown_id"].asInt();
        elems.fromJson(json["word_ids"]);
        E.init(nDim, nVSize);
        E.fromJson(json["e"]);
    }
};

template <typename ParamType>
class BatchedLookupNode;

template <typename ParamType>
class LookupNode : public Node, public Poolable<LookupNode<ParamType>> {
public:
    LookupNode() : Node("lookup") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void setParam(LookupTable<ParamType>& param) {
        param_ = param;
    }

    void setParam(ParamType &table) {
        param_ = &table;
    }

    void connect(Graph &graph, const vector<int> &ids) {
        ids_ = ids;
        graph.addNode(this);
    }

    PExecutor generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + "-" + addressToString(param_);
    }

    void compute() override {
        int dim = getDim() / ids_.size();
        int i = 0;
        for (int id : ids_) {
            Vec(val().v + i++ * dim, dim) = Vec(param_->val[id], dim);
        }
    }

    void backward() override {
        if (should_backward_) {
            int dim = getDim() / ids_.size();
            int i = 0;
            for (int id : ids_) {
                Vec(param_->grad[id], dim) += Vec(loss().v + i++ * dim, dim);
            }
        }
    }

    void setShouldBackward(bool should_backward) {
        should_backward_ = should_backward;
    }

private:
    ParamType* param_ = nullptr;
    vector<int> ids_;
    bool should_backward_ = true;

    friend class BatchedLookupNode<ParamType>;
};

namespace n3ldg_plus {

template <typename ParamType>
Node *embedding(Graph &graph,ParamType &lookup, const vector<int> ids,
        bool should_backward = true) {
    LookupNode<ParamType>* input_lookup =
        LookupNode<ParamType>::newNode(lookup.outDim() * ids.size());
    input_lookup->setShouldBackward(should_backward);
    input_lookup->setParam(lookup);
    input_lookup->connect(graph, ids);
    return input_lookup;
}

template <typename ParamType>
Node *embedding(Graph &graph,ParamType &lookup, int id, bool should_backward = true) {
    return embedding(graph, lookup, {id}, should_backward);
}

template <typename ParamType>
Node *embedding(Graph &graph, LookupTable<ParamType> &lookup, int dim, const vector<string> &words,
        bool should_backward = true) {
    vector<int> ids;
    ids.reserve(words.size());
    for (const string &word : words) {
        int id;
        if (!lookup.findElemId(word)) {
            if (lookup.nUNKId < 0) {
                cerr << "nUNKId is negative:" << lookup.nUNKId << endl;
                abort();
            }
            id = lookup.nUNKId;
        } else {
            id = lookup.getElemId(word);
        }
        ids.push_back(id);
    }
    LookupNode<ParamType>* input_lookup = LookupNode<ParamType>::newNode(dim * words.size());
    input_lookup->setParam(lookup.E);
    input_lookup->connect(graph, ids);
    input_lookup->setShouldBackward(should_backward);
    return input_lookup;
}

template <typename ParamType>
Node *embedding(Graph &graph, LookupTable<ParamType> &lookup, int dim, const string &word,
        bool should_backward = true) {
    return embedding(graph, lookup, dim, {word}, should_backward);
}

template <typename ParamType>
Node *embedding(Graph &graph, LookupTable<ParamType> &lookup, const vector<string> &words,
        bool should_backward = true) {
    return embedding(graph, lookup, lookup.nDim, words, should_backward);
}

}

template<typename ParamType>
#if USE_GPU
class LookupExecutor :public Executor {
public:
    int dim;
    ParamType *table;
    std::vector<int> xids;
    std::vector<int> backward_switches;

    void  forward() {
        int count = batch.size();
        xids.reserve(count);
        std::vector<dtype*> vals;
        vals.reserve(count);
        backward_switches.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            LookupNode<ParamType> *n = static_cast<LookupNode<ParamType>*>(batch[idx]);
            xids.push_back(n->xid);
            vals.push_back(n->val().value);
            backward_switches.push_back(n->should_backward ? 1 : 0);
        }

        n3ldg_cuda::LookupForward(xids, table->val.value, count, dim, vals);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val().verify("lookup forward"));
        }
#endif
    }

    void genericBackward(vector<dtype*> &losses);

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (Node *n : batch) {
            losses.push_back(n->loss().value);
        }
        genericBackward(losses);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        n3ldg_cuda::Assert(table->grad.verify("lookup backward grad"));
#endif
    }
};

template<>
void LookupExecutor<SparseParam>::genericBackward(vector<dtype*> &losses) {
        n3ldg_cuda::LookupBackward(xids, backward_switches,
                losses,
                batch.size(),
                dim,
                table->grad.value,
                table->dIndexers.value);
}

template<>
void LookupExecutor<Param>::genericBackward(vector<dtype*> &losses) {
        n3ldg_cuda::LookupBackward(xids, backward_switches,
                losses,
                batch.size(),
                dim,
                table->grad.value);
}

#else
class LookupExecutor :public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }
};
#endif

template<typename ParamType>
PExecutor LookupNode<ParamType>::generate() {
    LookupExecutor<ParamType>* exec = new LookupExecutor<ParamType>();
#if USE_GPU
    exec->table = param_;
    exec->dim = getDim();
#endif
    return exec;
}

#endif /*_LOOKUPTABLE_H*/
