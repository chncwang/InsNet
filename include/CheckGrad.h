#ifndef CHECKGREAD_H_
#define CHECKGREAD_H_

#include "MyLib.h"
#include <Eigen/Dense>
#include <functional>
#include "Node.h"
#include "BaseParam.h"

using namespace Eigen;

constexpr float CHECK_GRAD_STEP = 1e-3;

class CheckGrad {
public:
    vector<BaseParam*> _params;
    vector<string> _names;

    void init(const vector<BaseParam*> &params) {
        for (BaseParam *param : params) {
            _params.push_back(param);
            _names.push_back(param->getParamName());
        }
    }

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
    void check(Classifier* classifier, const vector<Example>& examples,
            const string& description) {
        dtype orginValue, plused_loss, minused_loss;
        int idx, idy;
        dtype mockGrad, computeGrad;
        for (int i = 0; i < _params.size(); i++) {
            _params[i]->randpoint(idx, idy);
            printf("%s, Checking gradient for %s[%d][%d]:\n", description.c_str(),
                    _names[i].c_str(), idx, idy);
            orginValue = _params[i]->val[idx][idy];

            _params[i]->val[idx][idy] = orginValue + CHECK_GRAD_STEP;
            plused_loss = 0.0;
            cout << "add 0.001" << endl;
            for (int j = 0; j < examples.size(); j++) {
                plused_loss += classifier->cost(examples[j]);
            }

            cout << "minus 0.001" << endl;
            _params[i]->val[idx][idy] = orginValue - CHECK_GRAD_STEP;
            minused_loss = 0.0;
            for (int j = 0; j < examples.size(); j++) {
                minused_loss += classifier->cost(examples[j]);
            }

            printf("plused_loss:%.10f, minused_loss:%.10f\n", plused_loss, minused_loss);

            mockGrad = (plused_loss - minused_loss) * 0.5 / CHECK_GRAD_STEP;
            mockGrad = mockGrad / examples.size();
            computeGrad = _params[i]->grad[idx][idy];

            printf("    mock grad = %.10f,\ncomputed grad = %.10f\n", mockGrad, computeGrad);

            _params[i]->val[idx][idy] = orginValue;
        }
    }
};

#endif /*CHECKGREAD_H_*/
