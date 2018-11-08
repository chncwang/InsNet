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
#include <boost/format.hpp>

using namespace Eigen;

class Serializable {
//    typedef nlohmann::json json;

//    std::unique_ptr<json> toJson() const;

//    void fromJson(const json &);
};

namespace n3ldg_cpu {

struct Tensor1D {
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
        dtype sum = 0.0f;
        for (int idx = 0; idx < dim; idx++) {
            os << v[idx] << std::endl;
            sum += v[idx];
        }
        os << sum << std::endl;
    }

    void load(std::istream &is) {
        int curDim;
        is >> curDim;
        init(curDim);
        dtype sum = 0.0f;
        for (int idx = 0; idx < dim; idx++) {
            is >> v[idx];
            sum += v[idx];
        }
        dtype saved_sum;
        is >> saved_sum;
        if (abs(saved_sum - sum) > 0.001) {
            std::cerr << boost::format(
                    "loading Tensor1D error, saved_sum is %1%, but computed sum is %2%")
                % saved_sum % sum << std::endl;
            abort();
        }
    }
};

struct Tensor2D {
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


    void save(std::ostream &os) const {
        os << size << " " << row << " " << col << std::endl;
        dtype sum = 0.0f;
        for (int idx = 0; idx < size; idx++) {
            os << v[idx] << std::endl;
            sum += v[idx];
        }
        os << sum << std::endl;
    }

    void load(std::istream &is) {
        int curSize, curRow, curCol;
        is >> curSize;
        is >> curRow;
        is >> curCol;
        init(curRow, curCol);
        dtype sum = 0.0f;
        for (int idx = 0; idx < size; idx++) {
            is >> v[idx];
            sum += v[idx];
        }
        dtype saved_sum;
        is >> saved_sum;
        if (abs(saved_sum - sum) > 0.001) {
            std::cerr << boost::format(
                    "loading Tensor2D error, saved_sum is %1%, but computed sum is %2%")
                % saved_sum % sum << std::endl;
            abort();
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
    virtual std::string name() const = 0;
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
#if USE_GPU
        copyFromHostToDevice();
#endif
    }

    bool verify(const char* message) {
#if TEST_CUDA
        return Verify(v, value, size, message);
#else
        return true;
#endif
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
