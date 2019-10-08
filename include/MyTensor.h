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
#include <iostream>
#include "MyTensor-def.h"

using namespace Eigen;


n3ldg_cpu::Tensor1D::Tensor1D() {
    dim = 0;
    v = NULL;
}

n3ldg_cpu::Tensor1D::~Tensor1D() {
    if (v) {
        delete[] v;
    }
}

void n3ldg_cpu::Tensor1D::init(int ndim) {
    dim = ndim;
    v = new dtype[dim];
    zero();
}

void n3ldg_cpu::Tensor1D::zero() {
    assert(v != NULL);
    for (int i = 0; i < dim; ++i) {
        v[i] = 0;
    }
}

std::string n3ldg_cpu::Tensor1D::toString() const {
    return toJson().toStyledString();
}

void n3ldg_cpu::Tensor1D::print() const {
    std::cout << toString() << std::endl;
}

const Mat n3ldg_cpu::Tensor1D::mat() const {
    return Mat(v, dim, 1);
}

Mat n3ldg_cpu::Tensor1D::mat() {
    return Mat(v, dim, 1);
}

const Mat n3ldg_cpu::Tensor1D::tmat() const {
    return Mat(v, 1, dim);
}

Mat n3ldg_cpu::Tensor1D::tmat() {
    return Mat(v, 1, dim);
}

const Vec n3ldg_cpu::Tensor1D::vec() const {
    return Vec(v, dim);
}

Vec n3ldg_cpu::Tensor1D::vec() {
    return Vec(v, dim);
}

dtype& n3ldg_cpu::Tensor1D::operator[](const int i) {
    if (i >= dim) {
        std::cerr << "i >= dim" << std::endl;
        abort();
    }
    return v[i];  // no boundary check?
}

const dtype& n3ldg_cpu::Tensor1D::operator[](const int i) const {
    assert(i < dim);
    return v[i];  // no boundary check?
}

n3ldg_cpu::Tensor1D& n3ldg_cpu::Tensor1D::operator=(const dtype &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a;
    return *this;
}

n3ldg_cpu::Tensor1D& n3ldg_cpu::Tensor1D::operator=(const std::vector<dtype> &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a[i];
    return *this;
}

n3ldg_cpu::Tensor1D& n3ldg_cpu::Tensor1D::operator=(const n3ldg_cpu::Tensor1D &a) { // assign a to every element
    for (int i = 0; i < dim; i++)
        v[i] = a[i];
    return *this;
}

void n3ldg_cpu::Tensor1D::random(dtype bound) {
    dtype min = -bound, max = bound;
    for (int i = 0; i < dim; i++) {
        v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }
}

Json::Value n3ldg_cpu::Tensor1D::toJson() const {
    Json::Value json;
    json["dim"] = dim;
    Json::Value json_arr;
    for (int i = 0; i < dim; ++i) {
        json_arr.append(v[i]);
    }
    json["value"] = json_arr;
    return json;
}

void n3ldg_cpu::Tensor1D::fromJson(const Json::Value &json) {
    dim = json["dim"].asInt();
    Json::Value json_arr = json["value"];
    for (int i = 0; i < dim; ++i) {
        v[i] = json_arr[i].asFloat();
    }
}

n3ldg_cpu::Tensor2D::Tensor2D() {
    col = row = 0;
    size = 0;
    v = NULL;
}

n3ldg_cpu::Tensor2D::~Tensor2D() {
    if (v) {
        delete[] v;
    }
    v = NULL;
    col = row = 0;
    size = 0;
}

void n3ldg_cpu::Tensor2D::init(int nrow, int ncol) {
    row = nrow;
    col = ncol;
    size = col * row;
    v = new dtype[size];
    zero();
}

void n3ldg_cpu::Tensor2D::zero() {
    assert(v != NULL);
    for (int i = 0; i < size; ++i) {
        v[i] = 0;
    }
}

std::string n3ldg_cpu::Tensor2D::toString() const {
    return toJson().toStyledString();
}

void n3ldg_cpu::Tensor2D::print() const {
    std::cout << toString() << std::endl;
}

const Mat n3ldg_cpu::Tensor2D::mat() const {
    return Mat(v, row, col);
}

Mat n3ldg_cpu::Tensor2D::mat() {
    return Mat(v, row, col);
}

const Vec n3ldg_cpu::Tensor2D::vec() const {
    return Vec(v, size);
}

Vec n3ldg_cpu::Tensor2D::vec() {
    return Vec(v, size);
}


dtype* n3ldg_cpu::Tensor2D::operator[](const int icol) {
    assert(icol < col);
    return &(v[icol*row]);  // no boundary check?
}

const dtype* n3ldg_cpu::Tensor2D::operator[](const int icol) const {
    assert(icol < col);
    return &(v[icol*row]);  // no boundary check?
}

void n3ldg_cpu::Tensor2D::assignAll(dtype a) {
    for (int i = 0; i < size; i++) {
        v[i] = a;
    }
}

n3ldg_cpu::Tensor2D& n3ldg_cpu::Tensor2D::operator=(const std::vector<dtype> &a) {
    for (int i = 0; i < size; i++)
        v[i] = a[i];
    return *this;
}

n3ldg_cpu::Tensor2D& n3ldg_cpu::Tensor2D::operator=(const std::vector<std::vector<dtype> > &a) {
    int offset = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            v[offset] = a[i][j];
            offset++;
        }
    }
    return *this;
}

n3ldg_cpu::Tensor2D& n3ldg_cpu::Tensor2D::operator=(const Tensor2D &a) {
    for (int i = 0; i < size; i++)
        v[i] = a.v[i];
    return *this;
}

void n3ldg_cpu::Tensor2D::random(dtype bound) {
    dtype min = -bound, max = bound;
    for (int i = 0; i < size; i++) {
        v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }
}

void n3ldg_cpu::Tensor2D::norm2one(dtype norm) {
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

Json::Value n3ldg_cpu::Tensor2D::toJson() const {
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

void n3ldg_cpu::Tensor2D::fromJson(const Json::Value &json) {
    row = json["row"].asInt();
    col = json["col"].asInt();
    Json::Value json_arr = json["value"];
    for (int i = 0; i < row * col; ++i) {
        v[i] = json_arr[i].asFloat();
    }
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
    n3ldg_cpu::Tensor1D::operator=(tensor);
    copyFromHostToDevice();
    return *this;
}

n3ldg_cuda::Tensor1D& n3ldg_cuda::Tensor1D::operator=(dtype v) {
    n3ldg_cpu::Tensor1D::operator=(v);
    copyFromHostToDevice();
    return *this;
}

void n3ldg_cuda::Tensor1D::random(dtype bound) {
    n3ldg_cpu::Tensor1D::random(bound);
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
    n3ldg_cpu::Tensor2D::zero();
}

void n3ldg_cuda::Tensor2D::random(dtype bound) {
    n3ldg_cpu::Tensor2D::random(bound);
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
    n3ldg_cpu::Tensor2D::assignAll(a);
    copyFromHostToDevice();
}

#endif

#endif
