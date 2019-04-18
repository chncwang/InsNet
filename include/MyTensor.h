#ifndef BasicTensor
#define BasicTensor

#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include "Def.h"
#include "serializable.h"
#include <boost/format.hpp>
#include <iostream>

using namespace Eigen;

namespace n3ldg_cpu {

struct Tensor1D : public N3LDGSerializable {
    dtype *v;
    int dim;

    Tensor1D() {
        dim = 0;
        v = NULL;
    }

    virtual ~Tensor1D() {
        if (v) {
            delete[] v;
        }
    }

    //please call this function before using it really. must! must! must!
    //only this function allocates memories
    virtual void init(int ndim) {
        dim = ndim;
        v = new dtype[dim];
        zero();
    }

    void zero() {
        assert(v != NULL);
        for (int i = 0; i < dim; ++i) {
            v[i] = 0;
        }
    }

    std::string toString() {
        return toJson().toStyledString();
    }

    const Mat mat() const {
        return Mat(v, dim, 1);
    }

    Mat mat() {
        return Mat(v, dim, 1);
    }

    const Mat tmat() const {
        return Mat(v, 1, dim);
    }

    Mat tmat() {
        return Mat(v, 1, dim);
    }

    const Vec vec() const {
        return Vec(v, dim);
    }

    Vec vec() {
        return Vec(v, dim);
    }

    dtype& operator[](const int i) {
        assert(i < dim);
        return v[i];  // no boundary check?
    }

    const dtype& operator[](const int i) const {
        assert(i < dim);
        return v[i];  // no boundary check?
    }

    Tensor1D& operator=(const dtype &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a;
        return *this;
    }

    Tensor1D& operator=(const std::vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    Tensor1D& operator=(const Tensor1D &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    virtual void random(dtype bound) {
        dtype min = -bound, max = bound;
        for (int i = 0; i < dim; i++) {
            v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
        }
    }

    virtual Json::Value toJson() const {
        Json::Value json;
        json["dim"] = dim;
        Json::Value json_arr;
        for (int i = 0; i < dim; ++i) {
            json_arr.append(v[i]);
        }
        json["value"] = json_arr;
        return json;
    }

    virtual void fromJson(const Json::Value &json) {
        dim = json["dim"].asInt();
        Json::Value json_arr = json["value"];
        for (int i = 0; i < dim; ++i) {
            v[i] = json_arr[i].asFloat();
        }
    }
};

struct Tensor2D : public N3LDGSerializable {
    dtype *v;
    int col, row, size;

    Tensor2D() {
        col = row = 0;
        size = 0;
        v = NULL;
    }

    virtual ~Tensor2D() {
        if (v) {
            delete[] v;
        }
        v = NULL;
        col = row = 0;
        size = 0;
    }

    //please call this function before using it really. must! must! must!
    //only this function allocates memories
    virtual void init(int nrow, int ncol) {
        row = nrow;
        col = ncol;
        size = col * row;
        v = new dtype[size];
        zero();
    }

    void zero() {
        assert(v != NULL);
        for (int i = 0; i < size; ++i) {
            v[i] = 0;
        }
    }

    const Mat mat() const {
        return Mat(v, row, col);
    }

    Mat mat() {
        return Mat(v, row, col);
    }

    const Vec vec() const {
        return Vec(v, size);
    }

    Vec vec() {
        return Vec(v, size);
    }


    //use it carefully, first col, then row, because rows are allocated successively
    dtype* operator[](const int icol) {
        assert(icol < col);
        return &(v[icol*row]);  // no boundary check?
    }

    const dtype* operator[](const int icol) const {
        assert(icol < col);
        return &(v[icol*row]);  // no boundary check?
    }

    virtual void assignAll(dtype a) {
        for (int i = 0; i < size; i++) {
            v[i] = a;
        }
    }

    Tensor2D& operator=(const std::vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < size; i++)
            v[i] = a[i];
        return *this;
    }

    Tensor2D& operator=(const std::vector<std::vector<dtype> > &a) { // assign a to every element
        int offset = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                v[offset] = a[i][j];
                offset++;
            }
        }
        return *this;
    }

    Tensor2D& operator=(const Tensor2D &a) { // assign a to every element
        for (int i = 0; i < size; i++)
            v[i] = a.v[i];
        return *this;
    }

    void random(dtype bound) {
        dtype min = -bound, max = bound;
        for (int i = 0; i < size; i++) {
            v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
        }
    }

    // for embeddings only, embedding matrix: vocabulary  * dim
    // each word's embedding is notmalized
    void norm2one(dtype norm = 1.0) {
        dtype sum;
        for (int idx = 0; idx < col; idx++) {
            sum = 0.000001;
            for (int idy = 0; idy < row; idy++) {
                sum += (*this)[idx][idy] * (*this)[idx][idy];
            }
            dtype scale = sqrt(norm / sum);
            for (int idy = 0; idy < row; idy++) {
                (*this)[idx][idy] *= scale;
            }
        }
    }

    virtual Json::Value toJson() const {
        Json::Value json;
        json["row"] = row;
        json["col"] = col;
        Json::Value json_arr;
        for (int i = 0; i < row * col; ++i) {
            json_arr.append(v[i]);
        }
        json["value"] = json_arr;
        return json;
    }

    virtual void fromJson(const Json::Value &json) {
        row = json["row"].asInt();
        col = json["col"].asInt();
        Json::Value json_arr = json["value"];
        for (int i = 0; i < row * col; ++i) {
            v[i] = json_arr[i].asFloat();
        }
    }
};

}

#if USE_GPU

namespace n3ldg_cuda {

class Transferable {
public:
    virtual void copyFromHostToDevice() = 0;
    virtual void copyFromDeviceToHost() = 0;
};

bool Verify(dtype *host, dtype* device, int len, const char* message);

struct Tensor1D : public n3ldg_cpu::Tensor1D, public Transferable {
    dtype *value = NULL;

    Tensor1D() = default;
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&) {
        abort();
    }
    void init(int len);
    void initOnMemoryAndDevice(int len);
    ~Tensor1D();

    virtual std::string name() const {
        return "Tensor1D";
    }

    Tensor1D& operator=(const Tensor1D &tensor) {
        n3ldg_cpu::Tensor1D::operator=(tensor);
        copyFromHostToDevice();
        return *this;
    }

    Tensor1D& operator=(dtype v) {
        n3ldg_cpu::Tensor1D::operator=(v);
        copyFromHostToDevice();
        return *this;
    }

    void random(dtype bound) {
        n3ldg_cpu::Tensor1D::random(bound);
        copyFromHostToDevice();
    }

    bool verify(const char *message) const {
#if TEST_CUDA
        return Verify(v, value, dim, message);
#else
        return true;
#endif
    }

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;

private:
    void initOnDevice(int len);
};

struct Tensor2D : public n3ldg_cpu::Tensor2D, public Transferable {
    dtype *value = NULL;

    Tensor2D() = default;
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&) {
        abort();
    }
    ~Tensor2D();

    void init(int row, int col);

    virtual std::string name() const {
        return "Tensor2D";
    }

    void zero() {
        assert(v != NULL);
        if (v == NULL) {
            std::cerr << "tensor2d v is null" << std::endl;
            abort();
        }
        n3ldg_cpu::Tensor2D::zero();
    }

    void random(dtype bound) {
        n3ldg_cpu::Tensor2D::random(bound);
        copyFromHostToDevice();
    }

    bool verify(const char* message) {
#if TEST_CUDA
        return Verify(v, value, size, message);
#else
        return true;
#endif
    }

    void assignAll(dtype a) {
        n3ldg_cpu::Tensor2D::assignAll(a);
        copyFromHostToDevice();
    }

    void initOnMemoryAndDevice(int row, int col);

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
private:
    void initOnDevice(int row, int col);
};


}
#endif

#endif
