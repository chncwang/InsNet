#ifndef N3LDG_PLUS_METRIC
#define N3LDG_PLUS_METRIC

#include <iostream>

namespace n3ldg_plus {

struct Metric {

  public:
    int overall_label_count;
    int correct_label_count;
    int predicated_label_count;

  public:
    Metric() {
        overall_label_count = 0;
        correct_label_count = 0;
        predicated_label_count = 0;
    }

    ~Metric() {}

    void reset() {
        overall_label_count = 0;
        correct_label_count = 0;
        predicated_label_count = 0;
    }

    void set(const Metric& other) {
        overall_label_count = other.overall_label_count;
        correct_label_count = other.correct_label_count;
        predicated_label_count = other.predicated_label_count;
    }

    bool bIdentical() {
        if (predicated_label_count == 0) {
            if (overall_label_count == correct_label_count) {
                return true;
            }
            return false;
        } else {
            if (overall_label_count == correct_label_count && predicated_label_count == correct_label_count) {
                return true;
            }
            return false;
        }
    }

    double getAccuracy() {
        if (overall_label_count + predicated_label_count == 0) return 1.0;
        if (predicated_label_count == 0) {
            return correct_label_count*1.0 / overall_label_count;
        } else {
            return correct_label_count*2.0 / (overall_label_count + predicated_label_count);
        }
    }


    void print() {
        if (predicated_label_count == 0) {
            std::cout << "Accuracy:\tP=" << correct_label_count << "/" << overall_label_count
                      << "=" << correct_label_count*1.0 / overall_label_count << std::endl;
        } else {
            std::cout << "Recall:\tP=" << correct_label_count << "/" << overall_label_count << "=" << correct_label_count*1.0 / overall_label_count
                      << ", " << "Accuracy:\tP=" << correct_label_count << "/" << predicated_label_count << "=" << correct_label_count*1.0 / predicated_label_count
                      << ", " << "Fmeasure:\t" << correct_label_count*2.0 / (overall_label_count + predicated_label_count) << std::endl;
        }
    }

  public:
    bool operator == (const Metric &a1) const {
        return correct_label_count == a1.correct_label_count && overall_label_count == a1.overall_label_count && predicated_label_count == a1.predicated_label_count;
    }

    bool operator != (const Metric &a1) const {
        return correct_label_count != a1.correct_label_count || overall_label_count != a1.overall_label_count || predicated_label_count != a1.predicated_label_count;
    }


};

}

#endif


