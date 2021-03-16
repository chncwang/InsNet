#ifndef _LOOKUPTABLE_H_
#define _LOOKUPTABLE_H_

#include "SparseParam.h"
#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "MatrixNode.h"
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

    template<typename Archive>
    void save(Archive &ar) const {
        ar(bFineTune, nDim, nVSize, nUNKId, elems, E);
    }

    template<typename Archive>
    void load(Archive &ar) {
        ar(bFineTune, nDim, nVSize, nUNKId, elems);
        E.init(nDim, nVSize);
        ar(E);
    }
};

template <typename ParamType>
class BatchedLookupNode;
template <typename ParamType>
class LookupExecutor;

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
        setColumn(ids.size());
        graph.addNode(this);
    }

    PExecutor generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + "-" + addressToString(param_) +
            (should_backward_ ? "-backward" : "nobackward");
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
    friend class LookupExecutor<ParamType>;
};

namespace n3ldg_plus {

template <typename ParamType>
Node *embedding(Graph &graph,ParamType &lookup, const vector<int> &ids,
        bool should_backward = true) {
    bool pool = ids.size() == 1;
    LookupNode<ParamType>* input_lookup =
        LookupNode<ParamType>::newNode(lookup.outDim() * ids.size(), pool);
    input_lookup->setIsPooled(pool);
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
    void  forward() override {
        int count = batch.size();
        vector<int> cols;
        cols.reserve(count);
        for (Node *node : batch) {
            LookupNode<ParamType> &l = dynamic_cast<LookupNode<ParamType> &>(*node);
            int col = l.getColumn();
//            cout << "LookupExecutor forward col:" << col << endl;
            cols.push_back(col);
        }
        max_col_ = *max_element(cols.begin(), cols.end());
        vector<int> ids, backward_switches;
        ids.reserve(count * max_col_);
        backward_switches.reserve(count);
        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            LookupNode<ParamType> &l = dynamic_cast<LookupNode<ParamType> &>(*node);
            for (int id : l.ids_) {
                ids.push_back(id);
            }
            for (int i = 0; i < max_col_ - l.ids_.size(); ++i) {
                ids.push_back(-1);
            }
            vals.push_back(l.getVal().value);
        }
        id_arr_.init(ids.data(), ids.size());
        col_arr_.init(cols.data(), count);
        int row = getRow();
        n3ldg_cuda::LookupForward(id_arr_.value, param().val.value, count, row, col_arr_.value,
                max_col_, vals);

#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        if (!shouldBackward()) {
            return;
        }

        int count = batch.size();
        std::vector<dtype*> grads;
        grads.reserve(count);
        for (Node *n : batch) {
            grads.push_back(n->loss().value);
        }
        genericBackward(grads);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        n3ldg_cuda::Assert(param().grad.verify("lookup backward grad"));
#endif
    }

private:
    void genericBackward(vector<dtype*> &);

    ParamType &param() {
        return *dynamic_cast<LookupNode<ParamType> &>(*batch.front()).param_;
    }

    bool shouldBackward() {
        return dynamic_cast<LookupNode<ParamType> &>(*batch.front()).should_backward_;
    }

    n3ldg_cuda::IntArray id_arr_, backward_switch_arr_, col_arr_;
    int max_col_;
};

template<>
void LookupExecutor<SparseParam>::genericBackward(vector<dtype*> &grads) {
        n3ldg_cuda::LookupBackward(id_arr_.value, grads, batch.size(), getRow(), col_arr_.value,
                max_col_, param().grad.value, param().dIndexers.value);
}

template<>
void LookupExecutor<Param>::genericBackward(vector<dtype*> &grads) {
        n3ldg_cuda::LookupBackward(id_arr_.value, grads, batch.size(), getRow(), col_arr_.value,
                max_col_, param().grad.value, nullptr);
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
    return new LookupExecutor<ParamType>();
}

#endif /*_LOOKUPTABLE_H*/
