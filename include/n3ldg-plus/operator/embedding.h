#ifndef N3LDG_PLUS_EMBEDDING_H
#define N3LDG_PLUS_EMBEDDING_H

#include "n3ldg-plus/param/sparse-param.h"
#include "n3ldg-plus/param/param.h"
#include "n3ldg-plus/nlp/vocab.h"
#include "n3ldg-plus/computation-graph/graph.h"
#include "n3ldg-plus/util/util.h"

namespace n3ldg_plus {

template<typename ParamType>
class Embedding : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    Vocab elems;
    ParamType E;
    bool bFineTune;
    int nDim;
    int nVSize;
    int nUNKId;
    bool inited = false;

    Embedding(const std::string &name = "embedding") : E(name) {
        nVSize = 0;
        nDim = 0;
        nUNKId = -1;
        bFineTune = false;
    }

    int size() const {
        return nVSize;
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&E};
    }
#endif

    void init(const Vocab &alpha, int dim, bool fineTune = true) {
        if (!inited) {
            elems = alpha;
            nVSize = elems.size();
            nUNKId = elems.from_string(UNKNOWN_WORD);
            initWeights(dim, fineTune);
            inited = true;
        }
    }

    void init(const Vocab &alpha, const std::string& inFile, bool fineTune = true,
            dtype norm = -1) {
        elems = alpha;
        nVSize = elems.size();
        nUNKId = elems.from_string(UNKNOWN_WORD);
        initWeights(inFile, fineTune, norm);
    }

    void initWeights(int dim, bool tune) {
        if (dim <=0 || nVSize == 0 || (nVSize == 1 && nUNKId >= 0)) {
            std::cerr << fmt::format("Embedding initWeights - dim:{} size:{}\n", dim, nVSize);
            std::cerr << "please check the alphabet" << std::endl;
            abort();
        }
        nDim = dim;
        std::cout << fmt::format("initWeights dim:{} vocabulary_size:{}\n", nDim, nVSize);
        E.init(nDim, nVSize);
        E.val().random(std::sqrt(1.0 / nDim));
        bFineTune = tune;
#if USE_GPU
        E.val().copyFromHostToDevice();
#endif
    }

    void initWeights(const std::string& inFile, bool tune, dtype norm = -1) {
        if (nVSize == 0 || (nVSize == 1 && nUNKId >= 0)) {
            std::cout << "nVSize:" << nVSize << " nUNKId:" << nUNKId << std::endl;
            std::cerr << "please check the alphabet" << std::endl;
            abort();
        }

        std::ifstream inf;
        inf.open(inFile.c_str());

        if (!inf.is_open()) {
            std::cerr << "please check the input file" << std::endl;
            abort();
        }

        std::string strLine, curWord;
        std::vector<std::string> sLines;
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
            std::cerr << "sLines size is 0" << std::endl;
            abort();
        }

        //find the first line, decide the wordDim;
        std::vector<std::string>::iterator it = sLines.begin();
        sLines.erase(it);
        std::vector<std::string> vecInfo;
        split_bychar(sLines[0], vecInfo, ' ');
        nDim = vecInfo.size() - 1;

        std::cout << fmt::format("nDim:{} nVSize:{}", nDim, nVSize);
        E.init(nDim, nVSize);

        std::cout << "word embedding dim is " << nDim << std::endl;

        bool bHasUnknown = false;
        std::unordered_set<int> indexers;
        nr::NRVec<dtype> sum(nDim);
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
                    E.val()[wordId][idy] += curValue;
                }
            }
        }

        if (count == 0) {
            E.val().random(std::sqrt(3.0 / nDim));
            std::cout << "find no overlapped lexicons in the embedding file" << std::endl;
            abort();
        }

        if (nUNKId >= 0 && !bHasUnknown) {
            for (int idx = 0; idx < nDim; idx++) {
                E.val()[nUNKId][idx] = sum[idx] / (count + 1);
            }
            indexers.insert(nUNKId);
            count++;
            std::cout << UNKNOWN_WORD << " not found, using averaged value to initialize." << std::endl;
        }


        int oovWords = 0;
        for (int id = 0; id < nVSize; id++) {
            if (indexers.find(id) == indexers.end()) {
                oovWords++;
                for (int idy = 0; idy < nDim; idy++) {
                    E.val()[id][idy] = nUNKId >= 0 ? E.val()[nUNKId][idy] : sum[idy] / (count + 1);
                }
            }
        }


        std::cout << "OOV num is " << oovWords << ", total num is " << nVSize << ", embedding oov ratio is " << oovWords * 1.0 / nVSize << std::endl;
        std::cout << "unknown id" << nUNKId << std::endl;
        bFineTune = tune;
        if (norm > 0) {
            E.val().norm2one(norm);
        }
#if USE_GPU
        E.val().copyFromHostToDevice();
#endif
    }

    std::vector<Tunable<BaseParam>*> tunableComponents() override {
        if (bFineTune) {
            return {&E};
        } else {
            return {};
        }
    }

    int getElemId(const std::string& strFeat) const {
        return elems.find_string(strFeat) ? elems.from_string(strFeat) : nUNKId;
    }

    bool findElemId(const std::string &str) const {
        return elems.find_string(str);
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

Node *embedding(Graph &graph, const std::vector<int> &ids, Param &lookup,
        bool should_backward = true);

Node *embedding(Graph &graph, int id, Param &lookup, bool should_backward = true);

Node *embedding(Graph &graph, const std::vector<std::string> &words, Embedding<Param> &lookup,
        bool should_backward = true);

Node *embedding(Graph &graph, const std::vector<std::string> &words,
        Embedding<SparseParam> &lookup,
        bool should_backward = true);

Node *embedding(Graph &graph, const std::string &word, Embedding<Param> &lookup,
        bool should_backward = true);

Node *embedding(Graph &graph, const std::string &word, Embedding<SparseParam> &lookup,
        bool should_backward = true);

}

#endif
