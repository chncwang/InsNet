#include "n3ldg-plus/base/tensor.h"
#include "eigen/Eigen/Dense"
#include "fmt/core.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"

using std::vector;
using std::string;
using std::to_string;
using std::cout;
using std::cerr;
using std::endl;
using std::default_random_engine;
using std::normal_distribution;

namespace n3ldg_plus {

cpu::Tensor1D::~Tensor1D() {
    if (v) {
        delete[] v;
    }
}

void cpu::Tensor1D::init(int ndim) {
    dim = ndim;
    v = new dtype[dim];
    zero();
}

void cpu::Tensor1D::zero() {
    assert(v != NULL);
    for (int i = 0; i < dim; ++i) {
        v[i] = 0;
    }
}

string cpu::Tensor1D::toString() const {
    string result = fmt::format("dim:{} ", dim);
    for (int i = 0; i < dim; ++i) {
        result += to_string(v[i]) + " ";
    }
    return result;
}

void cpu::Tensor1D::print() const {
    cout << toString() << endl;
}

const Mat cpu::Tensor1D::mat() const {
    return Mat(v, dim, 1);
}

Mat cpu::Tensor1D::mat() {
    return Mat(v, dim, 1);
}

const Mat cpu::Tensor1D::tmat() const {
    return Mat(v, 1, dim);
}

Mat cpu::Tensor1D::tmat() {
    return Mat(v, 1, dim);
}

const Vec cpu::Tensor1D::vec() const {
    return Vec(v, dim);
}

Vec cpu::Tensor1D::vec() {
    return Vec(v, dim);
}

dtype& cpu::Tensor1D::operator[](const int i) {
    if (i >= dim) {
        cerr << fmt::format("i >= dim i:{} dim:{}\n", i, dim);
        abort();
    }
    return v[i];  // no boundary check?
}

const dtype& cpu::Tensor1D::operator[](const int i) const {
    if (i >= dim) {
        cerr << fmt::format("i >= dim i:{} dim:{}\n", i, dim);
        abort();
    }
    return v[i];  // no boundary check?
}

cpu::Tensor1D& cpu::Tensor1D::operator=(const dtype &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a;
    return *this;
}

cpu::Tensor1D& cpu::Tensor1D::operator=(const vector<dtype> &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a[i];
    return *this;
}

cpu::Tensor1D& cpu::Tensor1D::operator=(const cpu::Tensor1D &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a[i];
    return *this;
}

void cpu::Tensor1D::random(dtype bound) {
    dtype min = -bound, max = bound;
    for (int i = 0; i < dim; i++) {
        v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }
}

vector<dtype> cpu::Tensor1D::toCpu() const {
    vector<dtype> result;
    result.resize(dim);
    memcpy(result.data(), v, sizeof(dtype) * dim);
    return result;
}

void cpu::Tensor1D::checkIsNumber() const {
    for (int i = 0; i < dim; ++i) {
        if (v[i] != v[i]) {
            cerr << "checkIsNumber - nan detected" << endl;
            abort();
        }
    }
}

cpu::Tensor2D::Tensor2D() {
    col = row = 0;
    size = 0;
    v = NULL;
}

cpu::Tensor2D::~Tensor2D() {
    if (v) {
        delete[] v;
    }
    v = NULL;
    col = row = 0;
    size = 0;
}

void cpu::Tensor2D::init(int nrow, int ncol) {
    row = nrow;
    col = ncol;
    size = col * row;
    v = new dtype[size];
    zero();
}

void cpu::Tensor2D::zero() {
    assert(v != NULL);
    for (int i = 0; i < size; ++i) {
        v[i] = 0;
    }
}

string cpu::Tensor2D::toString() const {
    string result = fmt::format("row:{} col:{} ", row, col);
    for (int i = 0; i < row * col; ++i) {
        result += to_string(v[i]) + " ";
    }
    return result;
}

void cpu::Tensor2D::print() const {
    cout << toString() << endl;
}

const Mat cpu::Tensor2D::mat() const {
    return Mat(v, row, col);
}

Mat cpu::Tensor2D::mat() {
    return Mat(v, row, col);
}

const Vec cpu::Tensor2D::vec() const {
    return Vec(v, size);
}

Vec cpu::Tensor2D::vec() {
    return Vec(v, size);
}


dtype* cpu::Tensor2D::operator[](const int icol) {
    assert(icol < col);
    return &(v[icol*row]);  // no boundary check?
}

const dtype* cpu::Tensor2D::operator[](const int icol) const {
    assert(icol < col);
    return &(v[icol*row]);  // no boundary check?
}

void cpu::Tensor2D::assignAll(dtype a) {
    for (int i = 0; i < size; i++) {
        v[i] = a;
    }
}

cpu::Tensor2D& cpu::Tensor2D::operator=(const vector<dtype> &a) {
    for (int i = 0; i < size; i++)
        v[i] = a[i];
    return *this;
}

cpu::Tensor2D& cpu::Tensor2D::operator=(const vector<vector<dtype> > &a) {
    int offset = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            v[offset] = a[i][j];
            offset++;
        }
    }
    return *this;
}

cpu::Tensor2D& cpu::Tensor2D::operator=(const Tensor2D &a) {
    for (int i = 0; i < size; i++)
        v[i] = a.v[i];
    return *this;
}

void cpu::Tensor2D::random(dtype bound) {
    dtype min = -bound, max = bound;
    for (int i = 0; i < size; i++) {
        v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }
}

void cpu::Tensor2D::randomNorm(dtype sd) {
    static default_random_engine eng(0);
    normal_distribution<> d(0, sd);
    for (int i = 0; i < size; i++) {
        v[i] = d(eng);
    }
}

void cpu::Tensor2D::norm2one(dtype norm) {
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

#if USE_GPU

cuda::Tensor1D::Tensor1D() = default;

cuda::Tensor1D::Tensor1D(Tensor1D &&) {
    abort();
}

string cuda::Tensor1D::name() const {
    return "Tensor1D";
}

cuda::Tensor1D& cuda::Tensor1D::operator=(const Tensor1D &tensor) {
    cpu::Tensor1D::operator=(tensor);
    copyFromHostToDevice();
    return *this;
}

cuda::Tensor1D& cuda::Tensor1D::operator=(dtype v) {
    cpu::Tensor1D::operator=(v);
    copyFromHostToDevice();
    return *this;
}

void cuda::Tensor1D::random(dtype bound) {
    cpu::Tensor1D::random(bound);
    copyFromHostToDevice();
}

bool cuda::Tensor1D::verify(const char *message) const {
#if TEST_CUDA
    return Verify(v, value, dim, message);
#else
    return true;
#endif
}

cuda::Tensor2D::Tensor2D() = default;

cuda::Tensor2D::Tensor2D(Tensor2D &&) {
    abort();
}

string cuda::Tensor2D::name() const {
    return "Tensor2D";
}

void cuda::Tensor2D::zero() {
    assert(v != NULL);
    if (v == NULL) {
        cerr << "tensor2d v is null" << endl;
        abort();
    }
    cpu::Tensor2D::zero();
}

void cuda::Tensor2D::random(dtype bound) {
    cpu::Tensor2D::random(bound);
    copyFromHostToDevice();
}

void cuda::Tensor2D::randomNorm(dtype sd) {
    cpu::Tensor2D::randomNorm(sd);
    copyFromHostToDevice();
}

bool cuda::Tensor2D::verify(const char* message) {
#if TEST_CUDA
    return Verify(v, value, size, message);
#else
    return true;
#endif
}

void cuda::Tensor2D::assignAll(dtype a) {
    cpu::Tensor2D::assignAll(a);
    copyFromHostToDevice();
}

#endif

}
