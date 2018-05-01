#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include "Def.h"

#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include <cmath>

namespace n3ldg_cuda {

struct NumberPointerArray {
    dtype **value = NULL;
    int len = 0;

    NumberPointerArray() = default;
    NumberPointerArray(NumberPointerArray&&) {
        abort();
    }
    NumberPointerArray(const NumberPointerArray &) {
        abort();
    }
    void init(dtype **host_arr, int len);
    ~NumberPointerArray();
};

struct PageLockedNumberPointerArray {
    dtype **value = NULL;
    int len = 0;

    PageLockedNumberPointerArray() = default;
    PageLockedNumberPointerArray(PageLockedNumberPointerArray&&) {
        abort();
    }
    PageLockedNumberPointerArray(const PageLockedNumberPointerArray &) {
        abort();
    }
    void init(dtype **host_arr, int len);
    ~PageLockedNumberPointerArray();

    dtype **GetDevicePointer() const;
};

struct NumberPointerPointerArray {
    dtype ***value = NULL;
    int len = 0;

    NumberPointerPointerArray() = default;
    NumberPointerPointerArray(NumberPointerPointerArray&&) {
        abort();
    }
    NumberPointerPointerArray(const NumberPointerPointerArray &) {
        abort();
    }
    void init(dtype ***host_arr, int len);
    ~NumberPointerPointerArray();
};

struct NumberArray {
    dtype *value = NULL;
    int len = 0;

    NumberArray() = default;
    NumberArray(NumberArray&&) {
        abort();
    }
    NumberArray(const NumberArray &) = delete;
    void init(dtype *host_arr, int len);
    void init(int len);
    ~NumberArray();
};

struct IntPointerArray {
    int **value = NULL;
    int len = 0;

    IntPointerArray() = default;
    IntPointerArray(IntPointerArray&&) {
        abort();
    }
    IntPointerArray(const IntPointerArray &) = delete;
    void init(int **host_arr, int len);
    ~IntPointerArray();
};

struct DeviceNumber {
    dtype *value = NULL;
    dtype v = 0.0f;

    DeviceNumber() = default;
    DeviceNumber(DeviceNumber &&) {
        abort();
    }
    DeviceNumber(const DeviceNumber&) {
        abort();
    }

    void init();
    void copyFromDeviceToHost();
    ~DeviceNumber();
};

struct DeviceInt {
    int *value = NULL;
    int v = 0;

    DeviceInt() = default;
    DeviceInt(DeviceInt &&) {
        abort();
    }
    DeviceInt(const DeviceInt&) {
        abort();
    }

    void init();
    void copyFromDeviceToHost();
    void copyFromHostToDevice();
    ~DeviceInt();
};

struct IntArray {
    int *value = NULL;
    int len = 0;

    IntArray() = default;
    IntArray(IntArray&&) {
        abort();
    }
    IntArray(const IntArray &) {
        abort();
    }
    void init(int *host_arr, int len);
    void init(int len);
    ~IntArray();
};

struct PageLockedIntArray {
    int *value = NULL;
    int len = 0;

    PageLockedIntArray() = default;
    PageLockedIntArray(PageLockedIntArray&&) {
        abort();
    }
    PageLockedIntArray(const PageLockedIntArray &) {
        abort();
    }
    void init(int *host_arr, int len);
    void init(int len);
    ~PageLockedIntArray();
};

struct BoolArray {
    bool *value = NULL;
    int len = 0;

    BoolArray() = default;
    BoolArray(BoolArray&&) {
        abort();
    }
    BoolArray(const BoolArray &) {
        abort();
    }
    void init(bool *host_arr, int len);
    void copyFromHost(bool *host_arr);
    void copyToHost(bool *host_arr);
    ~BoolArray();
};

bool Verify(dtype *host, dtype* device, int len, const char* message);
bool Verify(bool *host, bool *device, int len, const char* message);
bool Verify(int *host, int *device, int len, const char* message);

struct Tensor1D {
    dtype *value = NULL;
    dtype *v = NULL;
    int dim = 0;

    Tensor1D() = default;
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&) {
        abort();
    }
    void init(int len);
    ~Tensor1D();

    void save(std::ofstream &s) const {
    }

    void load(std::ifstream &s) {
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

    void zero() {
        assert(v != NULL);
        memset((void*)v, 0, dim * sizeof(dtype));;
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

    inline dtype& operator[](const int i) {
        return v[i];  // no boundary check?
    }

    inline const dtype& operator[](const int i) const {
        return v[i];  // no boundary check?
    }

    inline Tensor1D& operator=(const dtype &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a;
        return *this;
    }

    inline Tensor1D& operator=(const std::vector<dtype> &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline Tensor1D& operator=(const nr::NRVec<dtype> &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline Tensor1D& operator=(const Tensor1D &a) { // assign a to every element
        for (int i = 0; i < dim; i++)
            v[i] = a[i];
        return *this;
    }

    inline void random(dtype bound) {
        dtype min = -bound, max = bound;
        for (int i = 0; i < dim; i++) {
            v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
        }
    }

    bool verify(const char *message) {
#if TEST_CUDA
        return Verify(v, value, dim, message);
#else
        return true;
#endif
    }

    void copyFromHostToDevice();
    void copyFromDeviceToHost();
private:
    void initOnDevice(int len);
};

struct Tensor2D {
    dtype *value = NULL;
    dtype *v = NULL;
    int row = 0;
    int col = 0;
    int size = 0;

    Tensor2D() = default;
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&) {
        abort();
    }
    void init(int row, int col);
    ~Tensor2D();

    void save(std::ofstream &s) const {
    }

    void load(std::ifstream &s) {
    }

    // for embeddings only, embedding matrix: vocabulary  * dim
    // each word's embedding is notmalized
    inline void norm2one(dtype norm = 1.0) {
        dtype sum;
        for (int idx = 0; idx < row; idx++) {
            sum = 0.000001;
            for (int idy = 0; idy < col; idy++) {
                sum += (*this)[idx][idy] * (*this)[idx][idy];
            }
            dtype scale = sqrt(norm / sum);
            for (int idy = 0; idy < col; idy++) {
                (*this)[idx][idy] *= scale;
            }
        }
    }

    void zero() {
        assert(v != NULL);
        memset((void*)v, 0, row * col * sizeof(dtype));;
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
    dtype* operator[](const int irow) {
        return &(v[irow*col]);  // no boundary check?
    }

    const dtype* operator[](const int irow) const {
        return &(v[irow*col]);  // no boundary check?
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

    Tensor2D& operator=(const nr::NRMat<dtype> &a) { // assign a to every element
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

    // for embeddings only, embedding matrix: dim  * vocabulary
    // each word's embedding is notmalized
    void norm2one() {
        dtype sum;
        for (int idx = 0; idx < col; idx++) {
            sum = 0.000001;
            for (int idy = 0; idy < row; idy++) {
                sum += (*this)[idx][idy] * (*this)[idx][idy];
            }
            dtype scale = sqrt(sum);
            for (int idy = 0; idy < row; idy++) {
                (*this)[idx][idy] /= scale;
            }
        }
    }

    bool verify(const char* message) {
#if TEST_CUDA
        return Verify(v, value, size, message);
#else
        return true;
#endif
    }
    void initOnMemoryAndDevice(int row, int col);

    void copyFromHostToDevice();
    void copyFromDeviceToHost();
private:
    void initOnDevice(int row, int col);
};

void Assert(bool v);
void Memset(dtype *p, int len, dtype value);
void Memset(bool *p, int len, bool value);
void *Malloc(int size);
void Memcpy(void *dest, void *src, int size, cudaMemcpyKind kind);
void BatchMemset(const std::vector<dtype*> &vec, int count, int dim,
        dtype value);
void PrintNums(const dtype* p, int len);
void PrintInts(const int* p, int len);

void InitCuda();
void EndCuda();

void CopyFromOneVectorToMultiVals(const dtype *src, std::vector<dtype*> &vals,
        int count,
        int len);

enum ActivatedEnum {
    TANH,
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SELU
};

enum PoolingEnum {
    MAX,
    MIN,
    SUM,
    AVG
};

void Activated(ActivatedEnum activated, const dtype *src,
        const std::vector<dtype*>& dest,
        dtype* dest2,
        int len,
        bool is_being_trained,
        dtype drop_factor,
        const dtype *drop_mask);
void TanhForward(ActivatedEnum activated, const std::vector<dtype*> &xs,
        int count,
        int dim,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &ys);
void TanhBackward(ActivatedEnum activated, const std::vector<dtype*> &losses,
        const std::vector<dtype*> &vals,
        int count,
        int dim,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &in_losses);
void DropoutForward(const std::vector<dtype*> &xs, int count, int dim,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &ys);
void DropoutBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &vals,
        int count,
        int dim,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &in_losses);
void CopyForUniNodeForward(const std::vector<dtype*> &xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b);
void CopyForBiNodeForward(const std::vector<dtype*>& x1s,
        const std::vector<dtype *>& x2s,
        const dtype *b,
        dtype *x1s_dest,
        dtype *x2s_dest,
        dtype *b_dest,
        int count,
        int x1_len,
        int x2_len,
        int b_len);
void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col,
        int count,
        bool useb,
        bool should_x_transpose = false,
        bool should_W_transpose = false);


void CalculateLtyForUniBackward(ActivatedEnum activated,
        const std::vector<dtype*> &ly,
        const dtype *ty,
        const dtype *y,
        const dtype *drop_mask,
        dtype drop_factor,
        dtype *lty,
        int count,
        int dim);
void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(const dtype *lty,
        const dtype *lx, dtype *b, std::vector<dtype*> &losses, int count,
        int out_dim, int in_dim, bool use_b);
void AddLtyToParamBiasAndAddLxToInputLossesForBiBackward(const dtype *lty,
        const dtype *lx1,
        const dtype *lx2,
        dtype *b,
        std::vector<dtype*> &losses1,
        std::vector<dtype*> &losses2,
        int count,
        int out_dim,
        int in_dim1,
        int in_dim2);
void CalculateDropoutMask(dtype dropout_ratio, int count, int dim,
        dtype *mask);
void ConcatForward(const std::vector<dtype*> &in_vals,
        const std::vector<int> &in_dims,
        std::vector<dtype*> &vals,
        bool on_training,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int in_count,
        int out_dim);
void ConcatBackward(const std::vector<dtype*> &in_losses,
        const std::vector<int> &in_dims,
        std::vector<dtype*> &losses,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int in_count,
        int out_dim);
void LookupForward(const std::vector<int> &xids, const dtype *vocabulary,
        bool on_training,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        std::vector<dtype*> &vals);
void LookupBackward(const std::vector<int> &xids, int unknown_id,
        bool fine_tune,
        const std::vector<dtype*> &losses,
        const dtype *drop_mask,
        dtype drop_factor,
        int count,
        int dim,
        dtype *grad,
        bool *indexers);
void PoolForward(PoolingEnum pooling, const std::vector<dtype*> &in_vals,
        std::vector<dtype*> &vals,
        int count,
        const std::vector<int> &in_counts,
        int dim,
        int *hit_inputs);
void PoolBackward(const std::vector<dtype*> &losses,
        std::vector<dtype*> &in_losses,
        const std::vector<int> &in_counts,
        const int *hit_inputs,
        int count,
        int dim);
void SumPoolForward(PoolingEnum pooling, const std::vector<dtype*> &in_vals,
        int count,
        int dim,
        const std::vector<int> &in_counts,
        std::vector<dtype*> &vals);
void SumPoolBackward(PoolingEnum pooling, const std::vector<dtype*> &losses,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses);
void ScalarAttentionForward(const std::vector<dtype*> &ins,
        const std::vector<dtype*> &unnormeds,
        const std::vector<int> &in_counts, int count, int dim,
        std::vector<dtype*> &masks, std::vector<dtype*> &vals);
void ScalarAttentionBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals,
        const std::vector<dtype*> &masks,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses,
        std::vector<dtype*> &unnormed_losses);
void VectorAttentionForward(const std::vector<dtype*> &ins,
        const std::vector<dtype*> &unnormeds,
        const std::vector<int> &in_counts, int count, int dim,
        std::vector<dtype*> &masks, std::vector<dtype*> &vals);
void VectorAttentionBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals,
        const std::vector<dtype*> &masks,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses,
        std::vector<dtype*> &unnormed_losses);
void PMultiForward(const std::vector<dtype*> &ins1,
        const std::vector<dtype*> &ins2,
        int count,
        int dim,
        bool on_training,
        const dtype* drop_mask,
        dtype dropout,
        std::vector<dtype*> &vals);
void PMultiBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals1,
        const std::vector<dtype*> &in_vals2,
        int count,
        int dim,
        const dtype* drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &in_losses1,
        std::vector<dtype*> &in_losses2);
void PAddForward(const std::vector<std::vector<dtype*>> &ins, int count,
        int dim,
        int in_count,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &vals);
void PAddBackward(const std::vector<dtype*> &losses, int count, int dim,
        int in_count,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<std::vector<dtype*>> &in_losses);
void CalculateLyForLinearBackward(const std::vector<dtype*> &ly_vec, dtype *ly,
        int count, int dim);
void SoftMaxLoss(const std::vector<dtype*> &vals, std::vector<dtype*> &losses,
        int *correct_count,
        const std::vector<int> &answers,
        int batchsize,
        int count,
        int dim);
int Predict(const dtype* val, int dim);
dtype SquareSum(const dtype *v, int len);
dtype SquareSum(const dtype *v, const bool *indexers, int count, int dim);
void Rescale(dtype *v, int len, dtype scale);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        const bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps);
void *GraphHostAlloc();
}

#endif
