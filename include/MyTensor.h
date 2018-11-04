#ifndef BasicTensor

#define BasicTensor


#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <fstream>
#include "Def.h"

using namespace Eigen;

namespace n3ldg_cpu {

class Tensor1D {
public:
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

//    Tensor1D& operator=(const NRVec<dtype> &a) { // assign a to every element
//        for (int i = 0; i < dim; i++)
//            v[i] = a[i];
//        return *this;
//    }

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

    void save(std::ostream &os) const {
        os << dim << std::endl;
        os << v[0];
        for (int idx = 1; idx < dim; idx++) {
            os << " " << v[idx];
        }
        os << std::endl;
    }

    void load(std::istream &is) {
        int curDim;
        is >> curDim;
        init(curDim);
        for (int idx = 0; idx < dim; idx++) {
            is >> v[idx];
        }
    }

};


struct Tensor2D {
public:
    dtype *v;
    int col, row, size;

    Tensor2D() {
        col = row = 0;
        size = 0;
        v = NULL;
    }

    ~Tensor2D() {
        if (v) {
            delete[] v;
        }
        v = NULL;
        col = row = 0;
        size = 0;
    }

    //please call this function before using it really. must! must! must!
    //only this function allocates memories
    void init(int nrow, int ncol) {
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

    //use it carefully
    Tensor2D& operator=(const dtype &a) { // assign a to every element
        for (int i = 0; i < size; i++)
            v[i] = a;
        return *this;
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


    void save(std::ofstream &os) const {
        os << size << " " << row << " " << col << std::endl;
        os << v[0];
        for (int idx = 1; idx < size; idx++) {
            os << " " << v[idx];
        }
        os << std::endl;
    }

    void load(std::ifstream &is) {
        int curSize, curRow, curCol;
        is >> curSize;
        is >> curRow;
        is >> curCol;
        init(curRow, curCol);
        for (int idx = 0; idx < size; idx++) {
            is >> v[idx];
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

class Tensor1D : public n3ldg_cpu::Tensor1D, public Transferable {
public:
    dtype *value = NULL;

    Tensor1D() = default;
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&) {
        abort();
    }
    void init(int len);
    void initOnMemoryAndDevice(int len);
    ~Tensor1D();

    Tensor1D& operator=(const Tensor1D &tensor) {
        abort();
    }

    Tensor1D& operator=(dtype v) {
        abort();
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

}
#endif

#endif
