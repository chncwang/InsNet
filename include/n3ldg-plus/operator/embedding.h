#ifndef N3LDG_PLUS_EMBEDDING_H
#define N3LDG_PLUS_EMBEDDING_H

#include "n3ldg-plus/param/sparse-param.h"
#include "n3ldg-plus/param/param.h"
#include "n3ldg-plus/nlp/vocab.h"
#include "n3ldg-plus/computation-graph/graph.h"
#include "n3ldg-plus/util/util.h"

namespace n3ldg_plus {

class EmbeddingAbs {
public:
    virtual BaseParam &param() = 0;
};

template<typename ParamType>
class Embedding : public TunableCombination<BaseParam>, public EmbeddingAbs
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    Vocab vocab;
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

    BaseParam &param() override {
        return E;
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
            vocab = alpha;
            nVSize = vocab.size();
            nUNKId = vocab.from_string(UNKNOWN_WORD);
            initWeights(dim, fineTune);
            inited = true;
        }
    }

    void init(const Vocab &alpha, const std::string& inFile, bool fineTune = true,
            dtype norm = -1) {
        vocab = alpha;
        nVSize = vocab.size();
        nUNKId = vocab.from_string(UNKNOWN_WORD);
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
            if (vocab.find_string(curWord)) {
                int wordId = vocab.insert_string(curWord);
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
        return vocab.find_string(strFeat) ? vocab.from_string(strFeat) : nUNKId;
    }

    bool findElemId(const std::string &str) const {
        return vocab.find_string(str);
    }

    template<typename Archive>
    void save(Archive &ar) const {
        ar(bFineTune, nDim, nVSize, nUNKId, vocab, E);
    }

    template<typename Archive>
    void load(Archive &ar) {
        ar(bFineTune, nDim, nVSize, nUNKId, vocab);
        E.init(nDim, nVSize);
        ar(E);
    }
};

/// \ingroup operator
/// Find embeddings from a parameter matrix with the specified ids.
///
/// For example, assumming we have the parameter matrix param = [[0.1, 0.1], [0.2, 0.2]], embedding(graph, {1, 0}, param) will return [0.2, 0.2, 0.1, 0.1].
///
/// **The operators with the same parameter matrix will be executed in batch.**
/// For example, given the same *param*, embedding(graph, {0, 122, 3333, 33333, 1}, param) and embedding(graph, {0, 323, 34223, 1}, param) will be executed in batch.
/// \param graph The computation graph.
/// \param ids The column numbers of the parameter matrix to find the embeddings.
/// \param param The parameter matrix. If the gradients will(will not) be sparse, pass an instance of SparseParam(Param).
/// \param freeze Wether to freeze the parameter matrix.
/// \return The result tensor of concaternated found embeddings. Its size is equal to *param.row() * ids.size()*.
Node *embedding(Graph &graph, const std::vector<int> &ids, BaseParam &param, bool freeze = false);

Node *embedding(Graph &graph, int id, BaseParam &lookup, bool freeze = false);

Node *embedding(Graph &graph, const std::vector<std::string> &words, EmbeddingAbs &lookup,
        bool freeze = false);

Node *embedding(Graph &graph, const std::string &word, EmbeddingAbs &lookup,
        bool freeze = false);

}

#endif
