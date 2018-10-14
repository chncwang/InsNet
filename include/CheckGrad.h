#ifndef CHECKGREAD_H_
#define CHECKGREAD_H_

#include "MyLib.h"
#include <Eigen/Dense>
#include <functional>

using namespace Eigen;

class CheckGrad {

  public:
    vector<BaseParam*> _params;
    vector<string> _names;

  public:
    CheckGrad() {
        clear();
    }

    void clear() {
        _params.clear();
        _names.clear();
    }

    void add(BaseParam &param, const std::string &name) {
        this->add(&param, name);
    }

    void add(BaseParam* param, const string& name) {
        _params.push_back(param);
        _names.push_back(name);
    }

  public:
    template<typename Sample>
    struct Classifier {
        std::function<dtype(const Sample &sample)> loss;

        Classifier(const std::function<dtype(const Sample &sample)> &los) {
            loss = los;
        }

        dtype cost(const Sample &sample) {
            return loss(sample);
        }
    };

    template<typename Sample>
    void check(const std::function<dtype(const Sample &sample)> &loss,
            const std::vector<Sample> &samples,
            const std::string &description) {
        Classifier<Sample> classifier(loss);
        check(&classifier, samples, description);
    }

    template<typename Example, typename Classifier>
    void check(Classifier* classifier, const vector<Example>& examples, const string& description) {
        dtype orginValue, lossAdd, lossPlus;
        int idx, idy;
        dtype mockGrad, computeGrad;
        for (int i = 0; i < _params.size(); i++) {
            _params[i]->randpoint(idx, idy);
            orginValue = _params[i]->val[idx][idy];

            _params[i]->val[idx][idy] = orginValue + 0.001;
            lossAdd = 0.0;
            for (int j = 0; j < examples.size(); j++) {
                lossAdd += classifier->cost(examples[j]);
            }

            _params[i]->val[idx][idy] = orginValue - 0.001;
            lossPlus = 0.0;
            for (int j = 0; j < examples.size(); j++) {
                lossPlus += classifier->cost(examples[j]);
            }

            mockGrad = (lossAdd - lossPlus) / 0.002;
            mockGrad = mockGrad / examples.size();
            computeGrad = _params[i]->grad[idx][idy];


            printf("%s, Checking gradient for %s[%d][%d]:\t", description.c_str(),
                   _names[i].c_str(), idx, idy);
            printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

            _params[i]->val[idx][idy] = orginValue;
        }
    }

};



#endif /*CHECKGREAD_H_*/
