#ifndef BasicTensor
#define BasicTensor

#include "n3ldg-plus/base/def.h"
#include "n3ldg-plus/base/tensor-def.h"
#include "eigen/Eigen/Dense"
#include "fmt/core.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>

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

std::string cpu::Tensor1D::toString() const {
    std::string result = fmt::format("dim:{} ", dim);
    for (int i = 0; i < dim; ++i) {
        result += std::to_string(v[i]) + " ";
    }
    return result;
}

void cpu::Tensor1D::print() const {
    std::cout << toString() << std::endl;
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
        std::cerr << fmt::format("i >= dim i:{} dim:{}\n", i, dim);
        abort();
    }
    return v[i];  // no boundary check?
}

const dtype& cpu::Tensor1D::operator[](const int i) const {
    if (i >= dim) {
        std::cerr << fmt::format("i >= dim i:{} dim:{}\n", i, dim);
        abort();
    }
    return v[i];  // no boundary check?
}

cpu::Tensor1D& cpu::Tensor1D::operator=(const dtype &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a;
    return *this;
}

cpu::Tensor1D& cpu::Tensor1D::operator=(const std::vector<dtype> &a) { // assign a to every element
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

template<typename Archive>
void cpu::Tensor1D::serialize(Archive &ar) {
    ar(dim);
    ar(cereal::binary_data(v, dim * sizeof(dtype)));
}

std::vector<dtype> cpu::Tensor1D::toCpu() const {
    std::vector<dtype> result;
    result.resize(dim);
    memcpy(result.data(), v, sizeof(dtype) * dim);
    return result;
}

void cpu::Tensor1D::checkIsNumber() const {
    for (int i = 0; i < dim; ++i) {
        if (v[i] != v[i]) {
            std::cerr << "checkIsNumber - nan detected" << std::endl;
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

std::string cpu::Tensor2D::toString() const {
    std::string result = fmt::format("row:{} col:{} ", row, col);
    for (int i = 0; i < row * col; ++i) {
        result += std::to_string(v[i]) + " ";
    }
    return result;
}

void cpu::Tensor2D::print() const {
    std::cout << toString() << std::endl;
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

cpu::Tensor2D& cpu::Tensor2D::operator=(const std::vector<dtype> &a) {
    for (int i = 0; i < size; i++)
        v[i] = a[i];
    return *this;
}

cpu::Tensor2D& cpu::Tensor2D::operator=(const std::vector<std::vector<dtype> > &a) {
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
    static std::default_random_engine eng(0);
    std::normal_distribution<> d(0, sd);
    for (int i = 0; i < size; i++) {
        v[i] = d(eng);
        if (i < 100) {
            std::cout << v[i] << " ";
        }
    }
    std::cout << std::endl;
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

template<typename Archive>
void cpu::Tensor2D::serialize(Archive &ar) {
    ar(row);
    ar(col);
    ar(cereal::binary_data(v, row * col * sizeof(dtype)));
}

#if USE_GPU

n3ldg_cuda::Tensor1D::Tensor1D() = default;

n3ldg_cuda::Tensor1D::Tensor1D(Tensor1D &&) {
    abort();
}

std::string n3ldg_cuda::Tensor1D::name() const {
    return "Tensor1D";
}

n3ldg_cuda::Tensor1D& n3ldg_cuda::Tensor1D::operator=(const Tensor1D &tensor) {
    cpu::Tensor1D::operator=(tensor);
    copyFromHostToDevice();
    return *this;
}

n3ldg_cuda::Tensor1D& n3ldg_cuda::Tensor1D::operator=(dtype v) {
    cpu::Tensor1D::operator=(v);
    copyFromHostToDevice();
    return *this;
}

void n3ldg_cuda::Tensor1D::random(dtype bound) {
    cpu::Tensor1D::random(bound);
    copyFromHostToDevice();
}

bool n3ldg_cuda::Tensor1D::verify(const char *message) const {
#if TEST_CUDA
    return Verify(v, value, dim, message);
#else
    return true;
#endif
}

n3ldg_cuda::Tensor2D::Tensor2D() = default;

n3ldg_cuda::Tensor2D::Tensor2D(Tensor2D &&) {
    abort();
}

std::string n3ldg_cuda::Tensor2D::name() const {
    return "Tensor2D";
}

void n3ldg_cuda::Tensor2D::zero() {
    assert(v != NULL);
    if (v == NULL) {
        std::cerr << "tensor2d v is null" << std::endl;
        abort();
    }
    cpu::Tensor2D::zero();
}

void n3ldg_cuda::Tensor2D::random(dtype bound) {
    cpu::Tensor2D::random(bound);
    copyFromHostToDevice();
}

void n3ldg_cuda::Tensor2D::randomNorm(dtype sd) {
    cpu::Tensor2D::randomNorm(sd);
    copyFromHostToDevice();
}

bool n3ldg_cuda::Tensor2D::verify(const char* message) {
#if TEST_CUDA
    return Verify(v, value, size, message);
#else
    return true;
#endif
}

void n3ldg_cuda::Tensor2D::assignAll(dtype a) {
    cpu::Tensor2D::assignAll(a);
    copyFromHostToDevice();
}

#endif

}

#endif