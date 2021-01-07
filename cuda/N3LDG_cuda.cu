#include "N3LDG_cuda.h"
#include <array>
#include <boost/format.hpp>
#include <cstdlib>
#include <cstddef>
#include <functional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include "Printf_cuda.cuh"
#include "Printf_cuda.cu"
#include "Memory_cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cnmem.h"
#include <string>
#include <utility>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <numeric>
#include <memory>
#include "profiler.h"
#include "MyTensor-def.h"

namespace n3ldg_cuda {

using namespace std;
using boost::format;

#if USE_FLOAT
#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)
#define cuda_tanh(x) tanhf(x)
#define cuda_exp(x) __expf(x)
#define cuda_log(x) logf(x)
#else
#define cuda_sqrt(x) sqrt(x)
#define cuda_pow(x, y) pow(x, y)
#define cuda_tanh(x) tanh(x)
#define cuda_exp(x) exp(x)
#define cuda_log(x) log(x)
#endif

#define KERNEL_LOG

#ifdef KERNEL_LOG
#define  KernelPrintLine(format, ...)\
{\
    cuPrintf("block:x=%d,y=%d thread:x=%d,y=%d "#format"\n", blockIdx.x,\
            blockIdx.y, threadIdx.x, threadIdx.y,__VA_ARGS__);\
}
#else
#define KernelPrintLine(format, ...)
#endif

constexpr int TPB = 1024;
constexpr int BLOCK_COUNT = 56;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cerr << "cuda error:" << cudaGetErrorString(status) << endl;
        abort();
    }
}

void CheckCudaError() {
    //cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "cuda error:" << cudaGetErrorName(error) << endl;
        cerr << "cuda error:" << cudaGetErrorString(error) << endl;
        abort();
    }
}

void CallCnmem(cnmemStatus_t status) {
    assert(status == CNMEM_STATUS_SUCCESS);
}

void CallCublas(cublasStatus_t status) {
    assert(status == CUBLAS_STATUS_SUCCESS);
}

void CallCurand(curandStatus status) {
    assert(status == CURAND_STATUS_SUCCESS);
}

cublasHandle_t& GetCublasHandle() {
    static cublasHandle_t handle;
    static bool init;
    if (!init) {
        init = true;
        CallCublas(cublasCreate(&handle));
    }
    return handle;
}

cudaError_t MyCudaMemcpy(void *dest, void *src, size_t count,
        cudaMemcpyKind kind) {
    cudaError_t e;
    e = cudaMemcpyAsync(dest, src, count, kind);
    CallCuda(e);
    return e;
}

int NextTwoIntegerPowerNumber(int number) {
    int result = 1;
    while (number > result) {
        result <<= 1;
    }
    return result;
}

template <>
vector<bool> GPUArray<bool>::toCpu() const {
    bool *cpu_arr = new bool[len];
    CallCuda(MyCudaMemcpy(cpu_arr, value, sizeof(bool) * len, cudaMemcpyDeviceToHost));
    vector<bool> result;
    result.resize(len);
    for (int i = 0; i < len; ++i) {
        result.at(i) = cpu_arr[i];
    }
    delete[] cpu_arr;
    return result;
}

void DeviceInt::init() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, sizeof(int)));
}

void DeviceInt::copyFromDeviceToHost() {
    CallCuda(MyCudaMemcpy(&v, value, sizeof(int), cudaMemcpyDeviceToHost));
}

void DeviceInt::copyFromHostToDevice() {
    CallCuda(MyCudaMemcpy(value, &v, sizeof(int), cudaMemcpyHostToDevice));
}

DeviceInt::~DeviceInt() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void DeviceNumber::init() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, sizeof(int)));
}

void DeviceNumber::copyFromDeviceToHost() {
    CallCuda(MyCudaMemcpy(&v, value, sizeof(dtype), cudaMemcpyDeviceToHost));
}

DeviceNumber::~DeviceNumber() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void Tensor1D::init(int dim) {
    initOnDevice(dim);
#if TEST_CUDA
    v = new dtype[dim];
    zero();
#endif
}

void Tensor1D::initOnMemoryAndDevice(int dim) {
    initOnDevice(dim);
    v = new dtype[dim];
    zero();
}

void Tensor1D::initOnDevice(int dim) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, dim * sizeof(dtype)));
    this->dim = dim;
}

void Tensor1D::initOnMemory(int len) {
    v = new dtype[dim];
    zero();
}

Tensor1D::Tensor1D(const Tensor1D &t) {
    dim = t.dim;
    memcpy(v, t.v, dim *sizeof(dtype));
    CallCuda(MyCudaMemcpy(value, t.value, dim * sizeof(dtype), cudaMemcpyDeviceToDevice));
}

Tensor1D::~Tensor1D() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void Tensor1D::print() const {
    cout << "dim:" << dim << endl;
    PrintNums(value, dim);
}

void Tensor1D::copyFromHostToDevice() {
    assert(v != NULL);
    assert(value != NULL);
    CallCuda(MyCudaMemcpy(value, v, dim * sizeof(dtype), cudaMemcpyHostToDevice));
}

void Tensor1D::copyFromDeviceToHost() {
    if (v == nullptr) {
        initOnMemory(dim);
    }
    CallCuda(MyCudaMemcpy(v, value, dim * sizeof(dtype), cudaMemcpyDeviceToHost));
}

__device__ int DeviceDefaultIndex();
__device__ int DeviceDefaultStep();
int DefaultBlockCount(int len);

__global__ void KernelCheckIsNumber(dtype *v, int dim, int *error) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *error = 0;
    }
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim; i += step) {
        if (v[i] != v[i]) {
            *error = 1;
            return;
        }
    }
}

void CheckIsNumber(dtype *v, int dim) {
    int block_count = DefaultBlockCount(dim);
    DeviceInt error;
    error.init();
    KernelCheckIsNumber<<<block_count, TPB>>>(v, dim, error.value);
    CheckCudaError();
    error.copyFromDeviceToHost();
    if (error.v != 0) {
        cerr << "nan checked!" << endl;
        abort();
    }
}

void Tensor1D::checkIsNumber() const {
    n3ldg_cuda::CheckIsNumber(value, dim);
}

void Tensor2D::initOnMemoryAndDevice(int row, int col) {
    initOnDevice(row, col);
    v = new dtype[row * col];
    zero();
}

void Tensor2D::init(int row, int col) {
    initOnDevice(row, col);
#if TEST_CUDA
    v = new dtype[row * col];
    zero();
#endif
}

void Tensor2D::initOnDevice(int row, int col) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value,
                row * col * sizeof(dtype)));
    this->row = row;
    this->col = col;
    this->size = row * col;
}

Tensor2D::Tensor2D(const Tensor2D &t) {
    row = t.row;
    col = t.col;
    memcpy(v, t.v, sizeof(dtype) * row * col);
    CallCuda(MyCudaMemcpy(value, t.value, sizeof(dtype) * row * col,
                cudaMemcpyDeviceToDevice));
}

Tensor2D::~Tensor2D() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void Tensor2D::print() const {
    cout << "row:" << row << " col:" << col << endl;
    PrintNums(value, size);
}

void Tensor2D::copyFromHostToDevice() {
    CallCuda(MyCudaMemcpy(value, v, size * sizeof(dtype), cudaMemcpyHostToDevice));
}

void Tensor2D::copyFromDeviceToHost() {
    CallCuda(MyCudaMemcpy(v, value, size * sizeof(dtype), cudaMemcpyDeviceToHost));
}

void Assert(bool v, const string &message, const function<void(void)> &call) {
#if TEST_CUDA
    if (!v) {
        cerr << message << endl;
        call();
        abort();
    }
#endif
}

__device__ void DeviceAtomicAdd(dtype* address, dtype value) {
    float old = value;  
    float new_old;
    do {
        new_old = atomicExch(address, 0.0);
        new_old += old;
    } while ((old = atomicExch(address, new_old))!=0.0);
};

__device__ dtype cuda_dexp(dtype y) {
    return y;
}

__device__ dtype cuda_dtanh(dtype y) {
    return 1.0f - y * y;
}

__device__ dtype cuda_sigmoid(dtype x) {
    return 1.0f / (1.0f + cuda_exp(-x));
}

__device__ dtype cuda_dsigmoid(dtype y) {
    return y * (1.0f - y);
}

__device__ dtype cuda_relu(dtype x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ dtype cuda_drelu(dtype x) {
    return x > 0.0f ? 1 : 0.0f;
}

__device__ dtype cuda_leaky_relu(dtype x) {
    return x > 0.0f ? x : -0.1f * x;
}

__device__ dtype cuda_dleaky_relu(dtype x) {
    return x > 0.0f ? 1.0f : -0.1f;
}

__device__ dtype cuda_dsqrt(dtype y) {
    return 0.5 / y;
}

__device__ dtype SELU_LAMBDA = 1.0507009873554804934193349852946;
__device__ dtype SELU_ALPHA = 1.6732632423543772848170429916717;

__device__ dtype cuda_selu(dtype x) {
    return x <= 0.0f ? SELU_LAMBDA * SELU_ALPHA * (cuda_exp(x) - 1.0f) :
        SELU_LAMBDA * x;
}

__device__ dtype cuda_dselu(dtype x, dtype y) {
    return x <= 0.0f ? SELU_LAMBDA * SELU_ALPHA + y : SELU_LAMBDA;
}

void Random(dtype *v, int len, dtype bound) {
    dtype *mem = (dtype*)malloc(len * sizeof(dtype));
    assert(mem != NULL);
    dtype min = -bound, max = bound;
    for (int i = 0; i < len; i++) {
        mem[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }

    CallCuda(MyCudaMemcpy(v, mem, len * sizeof(dtype), cudaMemcpyHostToDevice));

    free(mem);
}

__device__ int DeviceDefaultIndex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int DeviceDefaultStep() {
    return gridDim.x * blockDim.x;
}

__device__ dtype DeviceAbs(dtype d) {
    return d > 0 ? d : -d;
}

int DefaultBlockCount(int len) {
    int block_count = (len - 1 + TPB) /
        TPB;
    return min(block_count, BLOCK_COUNT);
}

int DefaultBlockCountWithoutLimit(int len) {
    return (len - 1 + TPB) / TPB;
}

__global__ void KernelZero(dtype *v, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) {
        return;
    }
    v[index] = 0;
}

void Zero(dtype *v, int len) {
    int block_count = (len - 1 + TPB) /
        TPB;
    KernelZero<<<block_count, TPB>>>(v, len);
    CheckCudaError();
}

__global__ void PrintPointers(void **p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%p\n", p[i]);
    }
}

__global__ void KernelPrintNums(dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d %f\n", i, p[i]);
    }
}

void PrintNums(dtype* p, int len) {
    KernelPrintNums<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

__global__ void KernelPrintNums(dtype **p, int index, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d %f\n", i, p[index][i]);
    }
}

void PrintNums(dtype **p, int count_i, int len) {
    KernelPrintNums<<<1, 1>>>(p, count_i, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

__global__ void KernelPrintInts(int* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d\n", p[i]);
    }
}

void PrintInts(int* p, int len) {
    KernelPrintInts<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

void InitCuda(int device_id, float memory_in_gb) {
    cout << "device_id:" << device_id << endl;
    CallCuda(cudaSetDeviceFlags(cudaDeviceMapHost));

#if DEVICE_MEMORY == 0
    cnmemDevice_t device;
    device.size = 10000000000;
    device.device = device_id;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
#else
    CallCuda(cudaSetDevice(device_id));
#endif
    CallCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CallCuda(cudaPrintfInit());
    MemoryPool::Ins().Init(memory_in_gb);
}

void EndCuda() {
    cudaPrintfEnd();
    Profiler::Ins().Print();
}

__global__ void KernelCopyFromOneVectorToMultiVectors(dtype *src,
        dtype **dest, int count, int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[count_i][len_i] = src[i];
    }
}

void CopyFromOneVectorToMultiVals(dtype *src, vector<dtype*> &vals,
        int count,
        int len) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    int block_count = (len * count - 1 + TPB) / TPB;
    block_count = min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<block_count, TPB>>>(src,
            (dtype **)val_arr.value, count, len);
    CheckCudaError();
}

void CopyFromHostToDevice(vector<dtype*> &src,
        vector<dtype*> &dest, int count, int dim) {
    dtype *long_src = (dtype*)malloc(count * dim * sizeof(dtype));
    if (long_src == NULL) {
        cerr << "out of memory!" << endl;
        abort();
    }
    for (int i = 0; i < count; ++i) {
        memcpy(long_src + i * dim, src.at(i), dim * sizeof(dtype));
    }
    dtype *long_dest = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&long_dest,
                count * dim * sizeof(dtype*)));
    CallCuda(cudaMemcpy(long_dest, long_src, count * dim * sizeof(dtype*),
                cudaMemcpyHostToDevice));
    CopyFromOneVectorToMultiVals(long_dest, dest, count, dim);
    free(long_src);
    CallCuda(MemoryPool::Ins().Free(long_dest));
}

__global__ void KernelCopyFromMultiVectorsToOneVector(dtype **src, dtype *dest, int count,
        int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[i] = src[count_i][len_i];
    }
}

void CopyFromMultiVectorsToOneVector(vector<dtype*> &src,
        dtype *dest,
        int count,
        int len) {
    NumberPointerArray src_arr;
    src_arr.init((dtype**)src.data(), src.size());
    int block_count = DefaultBlockCount(len * count);
    KernelCopyFromMultiVectorsToOneVector<<<block_count, TPB>>>(
            (dtype**)src_arr.value, dest, count, len);
    CheckCudaError();
}

void CopyFromDeviceToHost(vector<dtype*> &src,
        vector<dtype*> &dest, int count, int dim) {
    dtype *long_src = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&long_src,
                count * dim * sizeof(dtype*)));
    CopyFromMultiVectorsToOneVector(src, long_src, count, dim);
    dtype *long_dest = (dtype*)malloc(count * dim * sizeof(dtype));
    if (long_dest == NULL) {
        cerr << "out of memory!" << endl;
        abort();
    }
    CallCuda(cudaMemcpy(long_dest, long_src, count * dim * sizeof(dtype),
                cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i) {
        memcpy(dest.at(i), long_dest + i * dim, dim * sizeof(dtype));
    }
    CallCuda(MemoryPool::Ins().Free(long_src));
    free(long_dest);
}

__global__ void KernelActivationForward(ActivatedEnum activated, dtype **xs,
        int count,
        int *dims,
        int max_dim,
        dtype **ys) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            if (activated == ActivatedEnum::TANH) {
                ys[count_i][dim_i] = cuda_tanh(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::SIGMOID) {
                ys[count_i][dim_i] = cuda_sigmoid(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::EXP) {
                ys[count_i][dim_i] = cuda_exp(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::RELU) {
                ys[count_i][dim_i] = cuda_relu(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::SQRT) {
                ys[count_i][dim_i] = cuda_sqrt(xs[count_i][dim_i]);
            } else {
                printf("KernelActivationForward - error enum\n");
                assert(false);
            }
        }
    }
}

void ActivationForward(ActivatedEnum activated, vector<dtype*> &xs,
        int count,
        vector<int> &dims,
        vector<dtype*> &ys) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray x_arr, y_arr;
    x_arr.init((dtype**)xs.data(), xs.size());
    y_arr.init((dtype**)ys.data(), ys.size());
    int block_count = DefaultBlockCount(count * max_dim);

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelActivationForward<<<block_count, TPB>>>(activated, (dtype* *)x_arr.value,
            count, dim_arr.value, max_dim, (dtype **)y_arr.value);
    CheckCudaError();
}

__global__ void KernelActivationBackward(ActivatedEnum activated,
        dtype **grads,
        dtype **vals,
        int count,
        int *dims,
        int max_dim,
        dtype** in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            dtype l;
            if (activated == ActivatedEnum::TANH) {
                l = cuda_dtanh(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::SIGMOID) {
                l = cuda_dsigmoid(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::EXP) {
                l = cuda_dexp(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::RELU) {
                l = cuda_drelu(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::SQRT) {
                l = cuda_dsqrt(vals[count_i][dim_i]);
            } else {
                printf("KernelActivationBackward - error enum\n");
                assert(false);
            }
            dtype v = l * grads[count_i][dim_i];
            DeviceAtomicAdd(in_grads[count_i] + dim_i, v);
        }
    }
}

void ActivationBackward(ActivatedEnum activated, vector<dtype*> &grads,
        vector<dtype*> &vals,
        int count,
        vector<int> &dims,
        vector<dtype*> &in_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray loss_arr, val_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    val_arr.init((dtype**)vals.data(), vals.size());
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    int block_count = DefaultBlockCount(count * max_dim);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelActivationBackward<<<block_count, TPB>>>(activated, loss_arr.value,
            val_arr.value, count, dim_arr.value, max_dim, (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelDropoutForward(dtype **xs, int count, int dim,
        bool is_training,
        dtype* drop_mask,
        dtype drop_factor,
        dtype **ys) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        if (is_training) {
            if (drop_mask[i] < drop_factor) {
                ys[count_i][dim_i] = 0.0f;
            } else {
                ys[count_i][dim_i] = xs[count_i][dim_i];
            }
        } else {
            ys[count_i][dim_i] = (1 - drop_factor) * xs[count_i][dim_i];
        }
    }
}

void DropoutForward(vector<dtype*> &xs, int count, int dim,
        bool is_training,
        dtype *drop_mask,
        dtype drop_factor,
        vector<dtype*> &ys) {
    if (drop_factor < 0 || drop_factor >= 1.0f) {
        cerr << "drop value is " << drop_factor << endl;
        abort();
    }
    NumberPointerArray x_arr, y_arr;
    x_arr.init((dtype**)xs.data(), xs.size());
    y_arr.init((dtype**)ys.data(), ys.size());
    int block_count = DefaultBlockCount(count * dim);
    KernelDropoutForward<<<block_count, TPB>>>(x_arr.value, count, dim, is_training, drop_mask,
            drop_factor, (dtype **)y_arr.value);
    CheckCudaError();
}

__global__ void KernelDropoutBackward(dtype **grads, dtype **vals,
        int count,
        int dim,
        bool is_training,
        dtype* drop_mask,
        dtype drop_factor,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        if (is_training) {
            if (drop_mask[i] >= drop_factor) {
                DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i]);
            }
        } else {
            DeviceAtomicAdd(in_grads[count_i] + dim_i,
                    (1 - drop_factor) * grads[count_i][dim_i]);
        }
    }
}

void DropoutBackward(vector<dtype*> &grads,
        vector<dtype*> &vals,
        int count,
        int dim,
        bool is_training,
        dtype *drop_mask,
        dtype drop_factor,
        vector<dtype*> &in_grads) {
    if (drop_factor < 0 || drop_factor >= 1) {
        cerr << "drop value is " << drop_factor << endl;
        abort();
    }
    NumberPointerArray loss_arr, val_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    val_arr.init((dtype**)vals.data(), vals.size());
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    int block_count = DefaultBlockCount(count * dim);
    KernelDropoutBackward<<<block_count, TPB>>>(loss_arr.value, val_arr.value, count, dim,
            is_training, drop_mask, drop_factor, (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelBucketForward(dtype *input, int count, int dim, dtype **ys) {
    int index = DeviceDefaultIndex();
    for (int i = index; i < count * dim; i+= DeviceDefaultStep()) {
        int count_i = i / dim;
        int dim_i = i % dim;
        ys[count_i][dim_i] = input[count_i * dim + dim_i];
    }
}

void BucketForward(vector<dtype> input, int count, int dim, vector<dtype*> &ys) {
    NumberArray input_arr;
    NumberPointerArray ys_arr;
    input_arr.init((dtype*)input.data(), input.size());
    ys_arr.init((dtype**)ys.data(), ys.size());
    int block_count = DefaultBlockCount(count * dim);
    KernelBucketForward<<<block_count, TPB>>>((dtype*)input_arr.value, count, dim,
            (dtype **)ys_arr.value);
    CheckCudaError();
}

__global__ void KernelCopyForUniNodeForward(dtype** xs, dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int x_total_len = count * x_len;
    int b_total_len = count * b_len;
    for (int i = index; i < x_total_len + b_total_len; i += step) {
        if (i < x_total_len) {
            int count_i = i / x_len;
            int len_i = i % x_len;
            xs_dest[i] = xs[count_i][len_i];
        } else if (use_b) {
            int b_i = i - x_total_len;
            int len_i = b_i % b_len;
            b_dest[b_i] = b[len_i];
        }
    }
}

void CopyForUniNodeForward(vector<dtype*> &xs, dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b) {
    NumberPointerArray x_arr;
    x_arr.init((dtype**)xs.data(), xs.size());
    int len = x_len + b_len;
    int block_count = min((count * len - 1 + TPB) / TPB, 56);
    KernelCopyForUniNodeForward<<<block_count, TPB>>>(
            (dtype**)x_arr.value, (dtype*)b, xs_dest, b_dest,
            count, x_len, b_len, use_b);
    CheckCudaError();
}

void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col,
        int count, bool useb, bool should_x_transpose,
        bool should_W_transpose) {
    cublasHandle_t &handle = GetCublasHandle();
    dtype alpha = 1;
    dtype beta = useb? 1 : 0;
    cublasOperation_t x_op = should_x_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldx = should_x_transpose ? count : col;
    cublasOperation_t W_op = should_W_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldw = should_W_transpose ? col : row;
#if USE_FLOAT
    CallCublas(cublasSgemm(handle, W_op, x_op, row, count, col,
                &alpha, W, ldw, x, ldx, &beta, y, row));
#else
    CallCublas(cublasDgemm(handle, W_op, x_op, row, count, col,
                &alpha, W, ldw, x, ldx, &beta, y, row));
#endif
}

__global__ void KernelVerify(dtype *host, dtype *device, int len,
        char *message, bool *success) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        dtype loss = host[index] - device[index];
        if (DeviceAbs(loss) > 0.001 && DeviceAbs(loss) > 0.001 * DeviceAbs(host[index])) {
            *success = false;
            KernelPrintLine("KernelVerify: host:%f device:%f loss:%f",
                    host[index],
                    device[index],
                    loss);
        }
    }
}

bool Verify(dtype *host, dtype *device, int len, const char* message) {
    NumberArray arr;
    arr.init(host, len);
    int block_count = DefaultBlockCount(len);
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(MyCudaMemcpy(m, (void *)message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool)));
    CallCuda(MyCudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    CallCuda(MyCudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    if (!success) {
        cerr << message << endl;
        abort();
    }
    return success;
}

__global__ void KernelVerify(bool *host, bool *device, int len,
        char *message, bool *success) {
    int index = DeviceDefaultIndex();
    if (index < len) {
        if (host[index] != device[index]) {
            *success = false;
            printf("KernelVerify %s: host:%d device:%d \n", message,
                    host[index],
                    device[index]);
            KernelPrintLine("KernelVerify: host:%d device:%d", host[index],
                    device[index]);
        }
    }
}

bool Verify(bool *host, bool *device, int len, const char* message) {
    BoolArray arr;
    arr.init(host, len);
    int block_count = (len + TPB - 1) / TPB;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(MyCudaMemcpy(m, (void *)message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool)));
    CallCuda(MyCudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    CallCuda(MyCudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

__global__ void KernelVerify(int *host, int *device, int len,
        char *message, bool *success) {
    int index = DeviceDefaultIndex();
    if (index < len) {
        if (host[index] != device[index]) {
            *success = false;
            printf("KernelVerify %s: host:%d device:%d \n", message,
                    host[index],
                    device[index]);
            KernelPrintLine("KernelVerify: host:%d device:%d", host[index],
                    device[index]);
        }
    }
}

bool Verify(int *host, int *device, int len, const char* message) {
    IntArray arr;
    arr.init(host, len);
    int block_count = (len + TPB - 1) / TPB;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(MyCudaMemcpy(m, (void*)message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, sizeof(bool)));
    CallCuda(MyCudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    CallCuda(MyCudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

constexpr int MAX_BLOCK_POWER = 100;

MemoryPool& MemoryPool::Ins() {
    static MemoryPool *p;
    if (p == NULL) {
        p = new MemoryPool;
        p->free_blocks_.resize(MAX_BLOCK_POWER + 1);
        p->busy_blocks_.reserve(10000);
    }
    return *p;
}

void appendFreeBlock(MemoryBlock &memory_block,
        vector<map<void*, MemoryBlock>> &free_blocks,
        int i,
        unordered_map<void*, MemoryBlock> &busy_blocks) {
    if (memory_block.size != (1 << i)) {
        cerr << boost::format("incorrect block size %1%, but i is %2%") % memory_block.size % i <<
            endl;
        abort();
    }
    free_blocks.at(i).insert(make_pair(memory_block.p, memory_block));
}

cudaError_t MemoryPool::Malloc(void **p, int size) {
    assert(*p == NULL);
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("Malloc");
#if DEVICE_MEMORY == 0
    CallCnmem(cnmemMalloc(p, size, NULL));
    profiler.EndEvent();
    return cudaSuccess;
#elif DEVICE_MEMORY == 1
    cudaError_t r = cudaMalloc(p, size);
    profiler.EndEvent();
    return r;
#else
    int fit_size = 1;
    int n = 0;
    while (fit_size < size) {
        fit_size <<= 1;
        ++n;
    }
    cudaError_t status = cudaErrorMemoryAllocation;
    while (status != cudaSuccess) {
        if (free_blocks_.at(n).empty()) {
            int higher_power = n + 1;
            while (higher_power <= MAX_BLOCK_POWER && free_blocks_.at(higher_power).empty()) {
                ++higher_power;
            }
            if (higher_power > MAX_BLOCK_POWER) {
                while (status != cudaSuccess) {
                    status = cudaMalloc(p, fit_size);
                    if (status != cudaSuccess) {
                        abort();
                    }
                }
                CallCuda(status);
                MemoryBlock block(*p, fit_size);
                busy_blocks_.insert(make_pair(*p, block));
            } else {
                auto &v = free_blocks_.at(higher_power);
                MemoryBlock &to_split = v.rbegin()->second;
                int half_size = to_split.size >> 1;
                void *half_address = static_cast<void*>(static_cast<char*>(to_split.p) +
                        half_size);
                MemoryBlock low_block(to_split.p, half_size, to_split.buddy),
                            high_block(half_address, half_size, to_split.p);
                v.erase(v.rbegin()->first);
                appendFreeBlock(low_block, free_blocks_, higher_power - 1, busy_blocks_);
                appendFreeBlock(high_block, free_blocks_, higher_power - 1, busy_blocks_);
            }
        } else {
            status = cudaSuccess;
            int this_size = free_blocks_.at(n).size();
            MemoryBlock &block = free_blocks_.at(n).rbegin()->second;
            *p = block.p;
            busy_blocks_.insert(make_pair(block.p, block));
            free_blocks_.at(n).erase(free_blocks_.at(n).rbegin()->first);
        }
    }
    profiler.EndEvent();

    return status;
#endif
}

pair<MemoryBlock *, MemoryBlock *> lowerAndhigherBlocks(MemoryBlock &a,
        MemoryBlock &b) {
    if (a.size != b.size) {
        cerr << "a.size is not equal to b.size" << endl;
        abort();
    }
    int distance = static_cast<char*>(a.p) - static_cast<char*>(b.p);
    if (distance == 0) {
        cerr << "block a and b has the same address" << endl;
        abort();
    }
    MemoryBlock &low = distance > 0 ? b : a;
    MemoryBlock &high = distance > 0 ? a : b;
    return make_pair(&low, &high);
}

bool isBuddies(MemoryBlock &a, MemoryBlock &b) {
    if (a.size != b.size) {
        return false;
    }
    auto pair = lowerAndhigherBlocks(a, b);
    return pair.second->buddy == pair.first->p &&
        ((char*)pair.second->p - (char*)pair.first->p) == a.size;
}

MemoryBlock mergeBlocks(MemoryBlock &a, MemoryBlock &b) {
    if (a.size != b.size) {
        cerr << "sizes of memory blocks to merge not equal" << endl;
        abort();
    }

    auto pair = lowerAndhigherBlocks(a, b);
    if ((char*)pair.second->p - (char*)pair.first->p != a.size ||
            (a.p != b.buddy && a.buddy != b.p)) {
        cerr << "a and b are not buddies" << endl;
        cerr << boost::format("a:%1%\nb:%2%") % a.toString() % b.toString() << endl;
        abort();
    }
    MemoryBlock block(pair.first->p, pair.first->size << 1, pair.first->buddy);
    return block;
}

void returnFreeBlock(MemoryBlock &block, vector<map<void*, MemoryBlock>> &free_blocks,
        int power,
        unordered_map<void*, MemoryBlock> &busy_blocks) {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("returnFreeBlock");
    MemoryBlock current_block = block;
    for (int i = power; i <= MAX_BLOCK_POWER; ++i) {
        map<void*, MemoryBlock> &v = free_blocks.at(i);
        void *free_p = (char*)current_block.p - (char*)current_block.buddy == current_block.size ?
            current_block.buddy : (void*)((char*)current_block.p + current_block.size);
        auto it = v.find(free_p);
        if (it == v.end() || (it->second.p != current_block.buddy &&
                    it->second.buddy != current_block.p)) {
            appendFreeBlock(current_block, free_blocks, i, busy_blocks);
            break;
        } else {
            MemoryBlock merged_block = mergeBlocks(it->second, current_block);
            current_block = merged_block;
            v.erase(it);
        }
    }
    profiler.EndEvent();
}

cudaError_t MemoryPool::Free(void *p) {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("Free");
#if DEVICE_MEMORY == 0
    CallCnmem(cnmemFree(p, NULL));
    profiler.EndEvent();
#elif DEVICE_MEMORY == 1
    cudaError_t r = cudaFree(p);
    profiler.EndEvent();
    return r;
#else
    auto it = busy_blocks_.find(p);
    if (it == busy_blocks_.end()) {
        cerr << "cannot find busy block " << p << endl;
        abort();
    }
    int size = it->second.size;
    int n = 0;
    while (size > 1) {
        size >>= 1;
        ++n;
    }
    if (it->second.size != (1 << n)) {
        cerr << boost::format("size:%1% n:%2%") % it->second.size % n << endl;
        abort();
    }

    auto block = it->second;
    busy_blocks_.erase(it);
    returnFreeBlock(block, free_blocks_, n, busy_blocks_);
    it = busy_blocks_.find(p);
    if (it != busy_blocks_.end()) {
        cerr << "can find erased block " << p << endl;
        abort();
    }

    profiler.EndEvent();
    if (busy_blocks_.find(p) != busy_blocks_.end()) {
        cerr << boost::format("Malloc - find freed p in busy blocks") << endl;
    }
    return cudaSuccess;
#endif
}

void Profiler::EndCudaEvent() {
    //cudaDeviceSynchronize();
    EndEvent();
}

__global__ void KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
        dtype *lty,
        dtype *lx,
        dtype *b,
        dtype **grads,
        int count,
        int out_dim,
        int in_dim,
        volatile dtype *block_sums,
        int *global_block_count,
        bool use_b) {
    __shared__ volatile dtype shared_arr[TPB];

    int count_i = blockIdx.y * blockDim.x + threadIdx.x;
    int dim_i = blockIdx.x;
    if (dim_i < out_dim) {
        if (use_b) {
            if (threadIdx.x == 0 && blockIdx.y == 0) {
                global_block_count[dim_i] = 0;
            }
            int lty_index = count_i * out_dim + dim_i;
            shared_arr[threadIdx.x] = count_i < count ? lty[lty_index] : 0.0f;
            __syncthreads();

            for (int i = (TPB >> 1); i > 0; i>>=1) {
                if (threadIdx.x < i) {
                    shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                block_sums[gridDim.y * blockIdx.x + blockIdx.y] =
                    shared_arr[0];
                if (atomicAdd(global_block_count + dim_i, 1) ==
                        gridDim.y - 1) {
                    dtype sum = 0.0;
                    for (int i = 0; i < gridDim.y; ++i) {
                        sum += block_sums[gridDim.y * blockIdx.x + i];
                    }
                    DeviceAtomicAdd(b + dim_i, sum);
                }
            }
        }
    } else {
        if (count_i < count) {
            dim_i -= out_dim;
            int lx_index = dim_i + count_i * in_dim;
            DeviceAtomicAdd(grads[count_i] + dim_i, lx[lx_index]);
        }
    }
}

void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(dtype *lty,
        dtype *lx, dtype *b, vector<dtype*> &grads, int count,
        int out_dim, int in_dim, bool use_b) {
    int block_y = (count - 1 + TPB) / TPB;
    dim3 block_dim(out_dim + in_dim, block_y, 1);
    NumberPointerArray loss_arr;
    loss_arr.init(grads.data(), count);
    Tensor1D block_sums;
    block_sums.init(block_y * out_dim);
    IntArray global_block_count_arr;
    global_block_count_arr.init(out_dim);
    KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward<<<block_dim,
        TPB>>>(lty, lx, b, (dtype **)loss_arr.value, count, out_dim, in_dim,
                block_sums.value, global_block_count_arr.value, use_b);
    CheckCudaError();
}

__global__ void KernelAddLtyToParamBiasAndAddLxToInputLossesForBiBackward(
        dtype *lty,
        dtype *lx1,
        dtype *lx2,
        dtype *b,
        dtype **grads1,
        dtype **grads2,
        int count,
        int out_dim,
        int in_dim1,
        int in_dim2,
        bool use_b,
        volatile dtype *block_sums,
        int *global_block_count) {
    __shared__ volatile dtype shared_arr[TPB];

    int count_i = blockIdx.y * blockDim.x + threadIdx.x;
    int dim_i = blockIdx.x;
    if (dim_i < out_dim) {
        if (threadIdx.x == 0 && blockIdx.y == 0) {
            global_block_count[dim_i] = 0;
        }
        //int lty_index = dim_i * count + count_i;
        int lty_index = dim_i + count_i * out_dim;
        shared_arr[threadIdx.x] = count_i < count ? lty[lty_index] : 0.0f;
        __syncthreads();

        for (int i = (TPB >> 1); i > 0; i>>=1) {
            if (threadIdx.x < i) {
                shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[gridDim.y * blockIdx.x + blockIdx.y] = shared_arr[0];
            if (atomicAdd(global_block_count + dim_i, 1) == gridDim.y - 1) {
                dtype sum = 0.0;
                for (int i = 0; i < gridDim.y; ++i) {
                    sum += block_sums[gridDim.y * blockIdx.x + i];
                }
                if (use_b) {
                    DeviceAtomicAdd(b + dim_i, sum);
                }
            }
        }
    } else if (dim_i < out_dim + in_dim1) {
        if (count_i < count) {
            dim_i -= out_dim;
            int lx_index = dim_i + count_i * in_dim1;
            DeviceAtomicAdd(grads1[count_i] + dim_i, lx1[lx_index]);
        }
    } else {
        if (count_i < count) {
            dim_i -= (out_dim + in_dim1);
            int lx_index = dim_i + count_i * in_dim2;
            DeviceAtomicAdd(grads2[count_i] + dim_i, lx2[lx_index]);
        }
    }
}

void AddLtyToParamBiasAndAddLxToInputLossesForBiBackward(dtype *lty,
        dtype *lx1,
        dtype *lx2,
        dtype *b,
        vector<dtype*> &grads1,
        vector<dtype*> &grads2,
        int count,
        int out_dim,
        int in_dim1,
        int in_dim2,
        bool use_b) {
    int block_y = (count - 1 + TPB) / TPB;
    dim3 block_dim(out_dim + in_dim1 + in_dim2, block_y, 1);
    NumberPointerArray loss1_arr;
    loss1_arr.init(grads1.data(), count);
    NumberPointerArray loss2_arr;
    loss2_arr.init(grads2.data(), count);
    Tensor1D block_sums;
    block_sums.init(block_y * out_dim);
    IntArray global_block_count_arr;
    global_block_count_arr.init(out_dim);
    KernelAddLtyToParamBiasAndAddLxToInputLossesForBiBackward<<<block_dim, TPB>>>(lty, lx1, lx2, b,
            (dtype **)loss1_arr.value, (dtype **)loss2_arr.value, count, out_dim,
            in_dim1, in_dim2, use_b, block_sums.value, global_block_count_arr.value);
    CheckCudaError();
}

constexpr int MAX_BATCH_COUNT = 1000000;

__global__ void KernelInitCurandStates(curandState_t *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = index; i < MAX_BATCH_COUNT; i += step) {
        curand_init(0, i, 0, &states[i]);
    }
}

curandState_t *GetCurandStates() {
    static curandState_t *states;
    if (states == NULL) {
        MemoryPool &pool = MemoryPool::Ins();
        CallCuda(pool.Malloc((void**)&states, sizeof(curandState_t) *
                    MAX_BATCH_COUNT));
        KernelInitCurandStates<<<BLOCK_COUNT, TPB>>>( states);
        CheckCudaError();
    }
    return states;
}

curandGenerator_t &GetGenerator() {
    static curandGenerator_t gen;
    static bool init;
    if (!init) {
        CallCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CallCurand(curandSetPseudoRandomGeneratorSeed(gen, 0));
        init = true;
    }
    return gen;
}

void CalculateDropoutMask(dtype drop_factor, int count, int dim, dtype* mask) {
    curandGenerator_t &gen = GetGenerator();
    CallCurand(curandGenerateUniform(gen, mask, count * dim));
}

__global__ void KernelConcatForward(dtype **ins, int *in_dims,
        dtype **outs,
        int count,
        int in_count,
        int out_dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        int in_dim_sum = 0;
        int last_in_dim_sum;
        int offset_j = 0;
        for (int j = 0; j < in_count; ++j) {
            last_in_dim_sum = in_dim_sum;
            in_dim_sum += in_dims[j];
            offset_j = j;
            if (out_dim_i < in_dim_sum) {
                break;
            }
        }
        int in_dim_i = out_dim_i - last_in_dim_sum;
        dtype v = ins[count_i * in_count + offset_j][in_dim_i];
        outs[count_i][out_dim_i] = v;
    }
}

void ConcatForward(vector<dtype*> &in_vals,
        vector<int> &in_dims,
        vector<dtype*> &vals,
        int count,
        int in_count,
        int out_dim) {
    int len = count * out_dim;
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    val_arr.init((dtype**)vals.data(), vals.size());
    IntArray in_dim_arr;
    in_dim_arr.init((int*)in_dims.data(), in_dims.size());

    KernelConcatForward<<<block_count, TPB>>>(in_val_arr.value,
            in_dim_arr.value, (dtype **)val_arr.value, count, in_count, out_dim);
    CheckCudaError();
}

__global__ void KernelConcatBackward(dtype **in_grads, int *in_dims,
        dtype **out_grads,
        int count,
        int in_count,
        int out_dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        int in_dim_sum = 0;
        int last_in_dim_sum;
        int offset_j = 0;
        for (int j = 0; j < in_count; ++j) {
            last_in_dim_sum = in_dim_sum;
            in_dim_sum += in_dims[j];
            offset_j = j;
            if (out_dim_i < in_dim_sum) {
                break;
            }
        }
        int in_dim_i = out_dim_i - last_in_dim_sum;
        DeviceAtomicAdd(in_grads[count_i * in_count + offset_j] +
                in_dim_i, out_grads[count_i][out_dim_i]);
    }
}

void ConcatBackward(vector<dtype*> &in_grads,
        vector<int> &in_dims,
        vector<dtype*> &grads,
        int count,
        int in_count,
        int out_dim) {
    int len = count * out_dim;
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);

    NumberPointerArray in_loss_arr, loss_arr;
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    loss_arr.init((dtype**)grads.data(), grads.size());
    IntArray in_dim_arr;
    in_dim_arr.init((int*)in_dims.data(), in_dims.size());

    KernelConcatBackward<<<block_count, TPB>>>((dtype **)in_loss_arr.value,
            in_dim_arr.value, loss_arr.value, count, in_count, out_dim);
    CheckCudaError();
}

__global__ void KernelScalarConcatForward(dtype **ins, int count,
        int *dims,
        int max_dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = ins[count_i * max_dim + dim_i][0];
        }
    }
}

void ScalarConcatForward(vector<dtype *> &ins, int count, vector<int> &dims,
        int max_dim,
        vector<dtype *> &results) {
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size());
    NumberPointerArray in_arr;
    in_arr.init((dtype**)ins.data(), ins.size());
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size());

    int block_count = DefaultBlockCount(count * max_dim);
    KernelScalarConcatForward<<<block_count, TPB>>>(in_arr.value, count, dim_arr.value,
            max_dim, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarConcatBackward(dtype **grads, int count, int *dims,
        int max_dim,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(input_grads[count_i * max_dim + dim_i], grads[count_i][dim_i]);
        }
    }
}

void ScalarConcatBackward(vector<dtype *> &grads, int count, vector<int> &dims,
        int max_dim,
        vector<dtype *> in_grads) {
    NumberPointerArray loss_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    in_loss_arr.init((dtype **)in_grads.data(), in_grads.size());
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScalarConcatBackward<<<block_count, TPB>>>(loss_arr.value, count, dim_arr.value,
            max_dim, (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelMemset(dtype *p, int len, dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(dtype *p, int len, dtype value) {
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
    CheckCudaError();
}

__global__ void KernelMemset(bool *p, int len, bool value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(bool *p, int len, bool value) {
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
    CheckCudaError();
}

void *Malloc(int size) {
    void *p;
    CallCuda(cudaMalloc(&p, size));
    return p;
}

__global__ void KernelBatchMemset(dtype **p, int count, int *dims, int max_dim,
        dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count ; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            p[count_i][dim_i] = value;
        }
    }
}

void BatchMemset(vector<dtype*> &vec, int count, vector<int> &dims, dtype value) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = (count * max_dim -1 + TPB) / TPB;
    block_count = min(block_count, BLOCK_COUNT);
    NumberPointerArray vec_arr;
    vec_arr.init((dtype**)vec.data(), vec.size());
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelBatchMemset<<<block_count, TPB>>>((dtype **)vec_arr.value, count, dim_arr.value,
            max_dim, value);
    CheckCudaError();
}

__global__ void KernelLookupForward(int *xids, dtype *vocabulary,
        int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (xid >= 0) {
            int voc_i = xid * dim + dim_i;
            vals[count_i][dim_i] = vocabulary[voc_i];
        } else {
            vals[count_i][dim_i] = 0.0f;
        }
    }
}

void LookupForward(vector<int> &xids, dtype *vocabulary,
        int count,
        int dim,
        vector<dtype*> &vals) {
    int block_count = min(BLOCK_COUNT, (count * dim - 1 + TPB) / TPB);
    IntArray xid_arr;
    xid_arr.init((int*)xids.data(), xids.size());
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    KernelLookupForward<<<block_count, TPB>>>(xid_arr.value, vocabulary,
            count, dim, const_cast<dtype**>(val_arr.value));
    CheckCudaError();
}

__global__ void KernelLookupBackward(int *xids, int *should_backward,
        dtype** grads,
        int count,
        int dim,
        dtype *grad,
        bool *indexers) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (should_backward[count_i]) {
            if (dim_i == 0) {
                indexers[xid] = true;
            }
            DeviceAtomicAdd(grad + xid * dim + dim_i, grads[count_i][dim_i]);
        }
    }
}

void LookupBackward(vector<int> &xids, vector<int> &should_backward,
        vector<dtype*> &grads,
        int count,
        int dim,
        dtype *grad,
        bool *indexers) {
    int block_count = min((count * dim - 1 + TPB) / TPB, BLOCK_COUNT);
    IntArray pl_arr;
    pl_arr.init((int*)xids.data(), xids.size());
    IntArray xid_arr;
    xid_arr.init((int*)pl_arr.value, xids.size());
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    IntArray should_backward_arr;
    should_backward_arr.init(should_backward.data(), should_backward.size());
    KernelLookupBackward<<<block_count, TPB>>>(
            const_cast<int *>(xid_arr.value),
            should_backward_arr.value,
            const_cast<dtype**>(loss_arr.value),
            count,
            dim,
            grad,
            indexers);
    CheckCudaError();
}

__global__ void KernelLookupBackward(int *xids, int *should_backward,
        dtype** grads,
        int count,
        int dim,
        dtype *grad) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (should_backward[count_i]) {
            DeviceAtomicAdd(grad + xid * dim + dim_i, grads[count_i][dim_i]);
        }
    }
}

void LookupBackward(vector<int> &xids, vector<int> &should_backward,
        vector<dtype*> &grads,
        int count,
        int dim,
        dtype *grad) {
    int block_count = min((count * dim - 1 + TPB) / TPB, BLOCK_COUNT);
    IntArray pl_arr;
    pl_arr.init((int*)xids.data(), xids.size());
    IntArray xid_arr;
    xid_arr.init((int*)pl_arr.value, xids.size());
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    IntArray should_backward_arr;
    should_backward_arr.init(should_backward.data(), should_backward.size());
    KernelLookupBackward<<<block_count, TPB>>>(
            const_cast<int *>(xid_arr.value),
            should_backward_arr.value,
            const_cast<dtype**>(loss_arr.value),
            count,
            dim,
            grad);
    CheckCudaError();
}

__global__ void KernelParamRowForward(dtype *param, int row_index, int param_row_count,
        int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int param_offset = dim_i * param_row_count + row_index;
        vals[count_i][dim_i] = param[param_offset];
    }
}

void ParamRowForward(dtype *param, int row_index, int param_row_count, int count, int dim,
        vector<dtype*> &vals) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    int block_count = DefaultBlockCount(count * dim);
    KernelParamRowForward<<<block_count, TPB>>>(param, row_index, param_row_count, count, dim,
            (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelPoolForward(PoolingEnum pooling, dtype **ins, int *in_counts,
        int max_in_count,
        dtype **outs,
        int count,
        int dim,
        int* hit_inputs) {
    __shared__ volatile extern dtype pool_shared_arr[];
    volatile dtype* shared_indexers = pool_shared_arr + blockDim.x;
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    if (in_count_i < in_count) {
        pool_shared_arr[threadIdx.x] = ins[batch_i * max_in_count +
            in_count_i][dim_i];
    } else {
        pool_shared_arr[threadIdx.x] = pooling == PoolingEnum::MAX ?
            -1e10 : 1e10;
    }
    shared_indexers[threadIdx.x] = threadIdx.x;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            if (pooling == PoolingEnum::MAX) {
                if (pool_shared_arr[threadIdx.x] < pool_shared_arr[plus_i]) {
                    pool_shared_arr[threadIdx.x] = pool_shared_arr[plus_i];
                    shared_indexers[threadIdx.x] = shared_indexers[plus_i];
                }
            } else {
                if (pool_shared_arr[threadIdx.x] > pool_shared_arr[plus_i]) {
                    pool_shared_arr[threadIdx.x] = pool_shared_arr[plus_i];
                    shared_indexers[threadIdx.x] = shared_indexers[plus_i];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hit_inputs[batch_i * dim + dim_i] = shared_indexers[0];
        outs[batch_i][dim_i] = pool_shared_arr[0];
    }
}

void PoolForward(PoolingEnum pooling, vector<dtype*> &in_vals, vector<dtype*> &vals,
        int count,
        vector<int> &in_counts,
        int dim,
        int *hit_inputs) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(dim, count, 1);

    NumberPointerArray in_val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());

    KernelPoolForward<<<block_dim, thread_count, thread_count * 2 *
        sizeof(dtype)>>>(pooling, in_val_arr.value, in_count_arr.value,
                max_in_count, (dtype **)val_arr.value, count, dim, hit_inputs);
    CheckCudaError();
}

__global__ void KernelPoolBackward(dtype ** grads, int *hit_inputs,
        int max_in_count,
        int count,
        int dim,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int input_i = hit_inputs[i];
        dtype loss = grads[count_i][dim_i];
        DeviceAtomicAdd(in_grads[count_i * max_in_count + input_i] + dim_i,
                loss);
    }
}

void PoolBackward(vector<dtype*> &grads, vector<dtype*> &in_grads,
        vector<int> &in_counts,
        int *hit_inputs,
        int count,
        int dim) {
    NumberPointerArray loss_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int block_count = (count * dim - 1 + TPB) / TPB;
    block_count = min(block_count, BLOCK_COUNT);
    KernelPoolBackward<<<block_count, TPB>>>((dtype**)loss_arr.value,
            hit_inputs,
            max_in_count,
            count,
            dim,
            (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSumPoolForward(PoolingEnum pooling, dtype **in_vals, int count,
        int dim,
        int *in_counts,
        int max_in_count,
        dtype **vals) {
    __shared__ volatile extern dtype pool_shared_arr[];
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    if (in_count_i < in_count) {
        pool_shared_arr[threadIdx.x] = in_vals[batch_i * max_in_count +
            in_count_i][dim_i];
    } else {
        pool_shared_arr[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            pool_shared_arr[threadIdx.x] += pool_shared_arr[plus_i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        vals[batch_i][dim_i] = pooling == PoolingEnum::SUM ?
            pool_shared_arr[0] : pool_shared_arr[0] / in_counts[batch_i];
    }
}

void SumPoolForward(PoolingEnum pooling, vector<dtype*> &in_vals, int count, int dim,
        vector<int> &in_counts,
        vector<dtype*> &vals) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(dim, count, 1);
    NumberPointerArray in_val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());

    KernelSumPoolForward<<<block_dim, thread_count,
        thread_count * sizeof(dtype)>>>(pooling, in_val_arr.value, count, dim, in_count_arr.value,
                max_in_count, (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelSumBackward(PoolingEnum pooling, dtype **grads,
        int *in_counts,
        int max_in_count,
        int count,
        int dim,
        dtype **in_grads) {
    int global_in_count_i = blockIdx.x * max_in_count + blockIdx.y;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (blockIdx.y < in_counts[blockIdx.x]) {
            DeviceAtomicAdd(in_grads[global_in_count_i] + i, pooling == PoolingEnum::SUM ?
                    grads[blockIdx.x][i] : grads[blockIdx.x][i] / in_counts[blockIdx.x]);
        }
    }
}

void SumPoolBackward(PoolingEnum pooling, vector<dtype*> &grads,
        vector<int> &in_counts,
        int count,
        int dim,
        vector<dtype*> &in_grads) {
    int thread_count = 8;
    while (thread_count < dim) {
        thread_count <<= 1;
    }
    thread_count = min(TPB, thread_count);

    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    dim3 block_dim(count, max_in_count, 1);
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    NumberPointerArray in_loss_arr;
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    KernelSumBackward<<<block_dim, thread_count>>>(pooling, loss_arr.value,
            (int*)in_count_arr.value, max_in_count, count, dim,
            (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixConcatForward(dtype **in_vals, int count, int in_dim, int *in_counts,
        int max_in_count,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_in_count * in_dim; i += step) {
        int max_in_dim_sum = max_in_count * in_dim;
        int count_i = i / max_in_dim_sum;
        int x = i % max_in_dim_sum;
        int in_count_i = x / in_dim;
        if (in_count_i < in_counts[count_i]) {
            int dim_i = x % in_dim;
            vals[count_i][x] = in_vals[count_i * max_in_count + in_count_i][dim_i];
        }
    }
}

void MatrixConcatForward(vector<dtype*> &in_vals, int count, int in_dim, vector<int> &in_counts,
        vector<dtype*> &vals) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int len = count * max_in_count * in_dim;
    int block_count = DefaultBlockCount(len);
    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init((dtype **)in_vals.data(), in_vals.size());
    val_arr.init((dtype **)vals.data(), vals.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());

    KernelMatrixConcatForward<<<block_count, TPB>>>((dtype**)in_val_arr.value, count, in_dim,
            in_count_arr.value, max_in_count, (dtype**)val_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixConcatBackward(dtype **grads, int count, int in_dim, int *in_counts,
        int max_in_count,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_in_count * in_dim; i += step) {
        int max_in_dim_sum = max_in_count * in_dim;
        int count_i = i / max_in_dim_sum;
        int x = i % max_in_dim_sum;
        int in_count_i = x / in_dim;
        if (in_count_i < in_counts[count_i]) {
            int dim_i = x % in_dim;
            DeviceAtomicAdd(in_grads[count_i * max_in_count + in_count_i] + dim_i,
                    grads[count_i][x]);
        }
    }
}

void MatrixConcatBackward(vector<dtype *> &grads, int count, int in_dim, vector<int> &in_counts,
        vector<dtype *> &in_grads) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int len = count * max_in_count * in_dim;
    int block_count = DefaultBlockCount(len);
    NumberPointerArray grad_arr, in_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    in_grad_arr.init((dtype **)in_grads.data(), in_grads.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    KernelMatrixConcatBackward<<<block_count, TPB>>>((dtype **)grad_arr.value, count, in_dim,
            in_count_arr.value, max_in_count, (dtype**)in_grad_arr.value);
}

__global__ void KernelMatrixAndVectorPointwiseMultiForward(dtype **matrix_vals,
        dtype **vector_vals,
        int count,
        int row,
        int *cols,
        int max_col,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * row * max_col; i += step) {
        int col_i = i / row % max_col;
        int row_i = i % row;
        int index_i = row * col_i + row_i;
        int count_i = i / (row * max_col);
        if (col_i < cols[count_i]) {
            vals[count_i][index_i] = matrix_vals[count_i][index_i] * vector_vals[count_i][row_i];
        }
    }
}

void MatrixAndVectorPointwiseMultiForward(vector<dtype *> &matrix_vals,
        vector<dtype *> &vector_vals,
        int count,
        int row,
        vector<int> &cols,
        vector<dtype *> &vals) {
    int max_col = *max_element(cols.begin(), cols.end());
    int block_count = DefaultBlockCount(count * max_col * row);
    NumberPointerArray matrix_arr, vector_arr, val_arr;
    matrix_arr.init((dtype **)matrix_vals.data(), count);
    vector_arr.init((dtype **)vector_vals.data(), count);
    val_arr.init((dtype **)vals.data(), count);
    IntArray col_arr;
    col_arr.init((int *)cols.data(), count);
    KernelMatrixAndVectorPointwiseMultiForward<<<block_count, TPB>>>((dtype **)matrix_arr.value,
            (dtype **)vector_arr.value, count, row, (int *)col_arr.value, max_col,
            (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixAndVectorPointwiseMultiBackwardForMatrix(dtype **grads,
        dtype **vector_vals,
        int count,
        int row,
        int *cols,
        int max_col,
        dtype **matrix_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * row * max_col; i += step) {
        int col_i = i / row % max_col;
        int row_i = i % row;
        int index_i = row * col_i + row_i;
        int count_i = i / (row * max_col);
        if (col_i < cols[count_i]) {
            dtype x = grads[count_i][index_i] * vector_vals[count_i][row_i]; 
            dtype *matrix_grads_addr = matrix_grads[count_i];
            DeviceAtomicAdd(matrix_grads_addr + index_i, x);
        }
    }
}

__global__ void KernelMatrixAndVectorPointwiseMultiBackwardForVector(dtype **grads,
        dtype **matrix_vals,
        int count,
        int row,
        int *cols,
        int max_col,
        dtype **vector_grads) {
    __shared__ volatile extern dtype shared_arr[];
    int count_i = blockIdx.y;
    int col = cols[count_i];
    int col_i = threadIdx.x;
    int row_i = blockIdx.x;
    if (col_i < col) {
        int index_i = row * col_i + row_i;
        shared_arr[threadIdx.x] = grads[count_i][index_i] * matrix_vals[count_i][index_i];
    } else {
        shared_arr[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            shared_arr[threadIdx.x] += shared_arr[plus_i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        vector_grads[count_i][row_i] = shared_arr[0];
    }
}

void MatrixAndVectorPointwiseMultiBackward(vector<dtype *> &grads, vector<dtype *> &matrix_vals,
        vector<dtype *> &vector_vals,
        int count,
        int row,
        vector<int> &cols,
        vector<dtype *> &matrix_grads,
        vector<dtype *> &vector_grads) {
    NumberPointerArray grad_arr, matrix_val_arr, vector_val_arr, matrix_grad_arr, vector_grad_arr;
    grad_arr.init((dtype **)grads.data(), count);
    matrix_val_arr.init((dtype **)matrix_vals.data(), count);
    vector_val_arr.init((dtype **)vector_vals.data(), count);
    matrix_grad_arr.init((dtype **)matrix_grads.data(), count);
    vector_grad_arr.init((dtype **)vector_grads.data(), count);
    IntArray col_arr;
    col_arr.init((int *)cols.data(), count);

    int max_col = *max_element(cols.begin(), cols.end());
    int block_count = DefaultBlockCount(count * max_col * row);
    KernelMatrixAndVectorPointwiseMultiBackwardForMatrix<<<block_count, TPB>>>(grad_arr.value,
            vector_val_arr.value, count, row, col_arr.value, max_col, matrix_grad_arr.value);
    CheckCudaError();

    int thread_count = NextTwoIntegerPowerNumber(max_col);
    dim3 block_dim(row, count, 1);
    KernelMatrixAndVectorPointwiseMultiBackwardForVector<<<block_dim, thread_count,
        thread_count * sizeof(dtype)>>>((dtype **)grad_arr.value, (dtype **)matrix_val_arr.value,
                count, row, (int *)col_arr.value, max_col, (dtype **)vector_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixColSumForward(dtype **in_vals, int count, int *cols, int max_col,
        int row,
        volatile dtype *block_sums,
        int *block_counters,
        dtype **vals) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x / max_col;
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int col_i = blockIdx.x % max_col;
    if (col_i >= cols[count_i]) {
        return;
    }
    int offset = row * col_i + row_i;
    shared_sum[threadIdx.x] = row_i < row ? in_vals[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            vals[count_i][col_i] = shared_sum[0];
        }
    }
}

void MatrixColSumForward(vector<dtype *> &in_vals, int count, vector<int> &cols, int row,
        vector<dtype *> &vals) {
    int thread_count = min(NextTwoIntegerPowerNumber(row), TPB);
    int block_y_count = (row - 1 + thread_count) / thread_count;
    int max_col = *max_element(cols.begin(), cols.end());
    dim3 block_dim(count * max_col, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count * max_col);
    IntArray block_counters;
    block_counters.init(count * max_col);

    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    val_arr.init((dtype **)vals.data(), vals.size());

    IntArray col_arr;
    col_arr.init(cols.data(), cols.size());

    KernelMatrixColSumForward<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            in_val_arr.value, count, col_arr.value, max_col, row, block_sums.value,
            block_counters.value, val_arr.value);
}

__global__ void KernelMatrixColSumBackward(dtype **grads, int count, int *cols, int max_col,
        int row,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_col * row; i += step) {
        int count_i = i / (max_col * row);
        int col_i = i / row % max_col;
        int offset = i % (max_col * row);

        if (col_i < cols[count_i]) {
            DeviceAtomicAdd(in_grads[count_i] + offset, grads[count_i][col_i]);
        }
    }
}

void MatrixColSumBackward(vector<dtype *> &grads, int count, vector<int> &cols, int row,
        vector<dtype *> &in_grads) {
    int max_col = *max_element(cols.begin(), cols.end());
    int block_count = DefaultBlockCount(count * max_col * row);
    NumberPointerArray grad_arr, in_grad_arr;
    grad_arr.init((dtype **)grads.data(), grads.size());
    in_grad_arr.init((dtype **)in_grads.data(), in_grads.size());
    IntArray col_arr;
    col_arr.init(cols.data(), cols.size());
    KernelMatrixColSumBackward<<<block_count, TPB>>>((dtype **)grad_arr.value, count,
            col_arr.value, max_col, row, in_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixAndVectorMultiForward(dtype **matrices, dtype **vectors, int count,
        int row,
        int *cols,
        int max_col,
        dtype **vals) {
    __shared__ volatile extern dtype shared_arr[];
    int count_i = blockIdx.y;
    int col = cols[count_i];
    int row_i = blockIdx.x;
    int matrix_offset = row * threadIdx.x + row_i;
    if (threadIdx.x < col) {
        shared_arr[threadIdx.x] = matrices[count_i][matrix_offset] * vectors[count_i][threadIdx.x];
    } else {
        shared_arr[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            int ii = threadIdx.x + i;
            shared_arr[threadIdx.x] += shared_arr[ii];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        vals[count_i][row_i] = shared_arr[0];
    }
}

void MatrixAndVectorMultiForward(vector<dtype *> &matrices, vector<dtype *> &vectors, int count,
        int row,
        vector<int> &cols,
        vector<dtype *> &vals) {
    NumberPointerArray matrix_arr, vector_arr, val_arr;
    matrix_arr.init((dtype **)matrices.data(), count);
    vector_arr.init((dtype **)vectors.data(), count);
    val_arr.init((dtype **)vals.data(), count);
    IntArray col_arr;
    col_arr.init(cols.data(), count);
    int max_col = *max_element(cols.begin(), cols.end());
    int thread_count = NextTwoIntegerPowerNumber(max_col);
    dim3 block_dim(row, count, 1);
    KernelMatrixAndVectorMultiForward<<<block_dim, thread_count,
        thread_count * sizeof(dtype)>>>(matrix_arr.value, vector_arr.value,
                count, row, col_arr.value, max_col, val_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixAndVectorMultiBackwardForMatrix(dtype **grads, dtype **vectors,
        int count,
        int row,
        int *cols,
        int max_col,
        dtype **matrix_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_col * row; i += step) {
        int count_i = i / (max_col * row); 
        int col_i = i / row % max_col;
        int row_i = i % row;
        int offset = i % (max_col * row);
        if (col_i < cols[count_i]) {
            DeviceAtomicAdd(matrix_grads[count_i] + offset,
                    grads[count_i][row_i] * vectors[count_i][col_i]);
        }
    }
}

__global__ void KernelMatrixAndVectorMultiBackwardForVector(dtype **grads, dtype **matrices,
        volatile dtype *block_sums,
        int *block_counters,
        int count,
        int row,
        int *cols,
        int max_col,
        dtype **vector_grads) {
    __shared__ volatile dtype shared_arr[TPB];
    __shared__ volatile bool is_last_block;
    if (blockIdx.z >= cols[blockIdx.x]) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x * max_col + blockIdx.z] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int matrix_offset = blockIdx.z * row + row_i;
    shared_arr[threadIdx.x] = row_i < row ?
        matrices[blockIdx.x][matrix_offset] * grads[blockIdx.x][row_i] : 0;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>=1) {
        if (threadIdx.x < i) {
            shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * max_col * gridDim.y +
        blockIdx.z * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_arr[0];
        if (atomicAdd(block_counters + max_col * blockIdx.x + blockIdx.z, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * max_col * gridDim.y + blockIdx.z * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_arr[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>=1) {
            if (threadIdx.x < i) {
                shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            DeviceAtomicAdd(vector_grads[blockIdx.x] + blockIdx.z, shared_arr[0]);
        }
    }
}

void MatrixAndVectorMultiBackward(vector<dtype *> &grads, vector<dtype *> &matrices,
        vector<dtype *> &vectors,
        int count,
        int row,
        vector<int> &cols,
        vector<dtype *> &matrix_grads,
        vector<dtype *> &vector_grads) {
    int max_col = *max_element(cols.begin(), cols.end());
    NumberPointerArray grad_arr, vector_arr, matrix_arr, matrix_grad_arr, vector_grad_arr;
    grad_arr.init(grads.data(), count);
    vector_arr.init(vectors.data(), count);
    matrix_arr.init(matrices.data(), count);
    matrix_grad_arr.init(matrix_grads.data(), count);
    vector_grad_arr.init(vector_grads.data(), count);
    IntArray col_arr;
    col_arr.init(cols.data(), count);

    int block_count = DefaultBlockCount(count * max_col * row);
    KernelMatrixAndVectorMultiBackwardForMatrix<<<block_count, TPB>>>(grad_arr.value,
            vector_arr.value, count, row, col_arr.value, max_col, matrix_grad_arr.value);
    CheckCudaError();

    int y = (row - 1) / TPB + 1;
    dim3 block_dim(count, y, max_col);
    NumberArray block_sums;
    block_sums.init(y * max_col * count);
    IntArray block_counters;
    block_counters.init(max_col * count);

    KernelMatrixAndVectorMultiBackwardForVector<<<block_dim, TPB>>>(grad_arr.value,
            matrix_arr.value, block_sums.value, block_counters.value, count, row,
            col_arr.value, max_col, vector_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixTransposeForward(dtype **matrices, int count, int input_row,
        int *input_cols,
        int max_col,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int max_dim = input_row * max_col;
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        int input_col = input_cols[count_i];
        if (dim_i < input_col * input_row) {
            int input_col_i = dim_i % input_col;
            int input_row_i = dim_i / input_col;
            vals[count_i][dim_i] = matrices[count_i][input_col_i * input_row + input_row_i];
        }
    }
}

void MatrixTransposeForward(vector<dtype *> &matrices, int count, int input_row,
        vector<int> &input_cols,
        vector<dtype *> &vals) {
    NumberPointerArray matrix_arr, val_arr;
    matrix_arr.init((dtype **)matrices.data(), count);
    val_arr.init((dtype **)vals.data(), count);
    IntArray input_col_arr;
    input_col_arr.init(input_cols.data(), count);
    int max_col = *max_element(input_cols.begin(), input_cols.end());
    int block_count = DefaultBlockCount(count * input_row * max_col);
    KernelMatrixTransposeForward<<<block_count, TPB>>>(matrix_arr.value, count, input_row,
            input_col_arr.value, max_col, val_arr.value);
    CheckCudaError();
}

__global__ void KernelPMultiForward(dtype **ins1, dtype **ins2, int count, int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        vals[count_i][dim_i] = ins1[count_i][dim_i] * ins2[count_i][dim_i];
    }
}

void PMultiForward(vector<dtype*> &ins1, vector<dtype*> &ins2, int count, int dim,
        vector<dtype*> &vals) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray ins1_arr, ins2_arr, val_arr;
    ins1_arr.init((dtype**)ins1.data(), count);
    ins2_arr.init((dtype**)ins2.data(), count);
    val_arr.init((dtype**)vals.data(), count);
    KernelPMultiForward<<<block_count, TPB>>>(ins1_arr.value, ins2_arr.value, count, dim,
            (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelDivForward(dtype **numerators, dtype **denominators,
        int count,
        int *dims,
        int max_dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = numerators[count_i][dim_i] / denominators[count_i][0];
        }
    }
}

void DivForward(vector<dtype*> numerators, vector<dtype*> denominators,
        int count,
        vector<int> &dims,
        vector<dtype*> &results) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray numerator_arr, denominator_arr, result_arr;
    numerator_arr.init((dtype**)numerators.data(), count);
    denominator_arr.init((dtype**)denominators.data(), count);
    result_arr.init((dtype**)results.data(), count);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelDivForward<<<block_count, TPB>>>(numerator_arr.value, denominator_arr.value, count,
            dim_arr.value, max_dim, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelDivNumeratorBackward(dtype **grads,
        dtype **denominator_vals,
        int count,
        int *dims,
        int max_dim,
        dtype **numerator_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(numerator_grads[count_i] + dim_i, grads[count_i][dim_i] /
                    denominator_vals[count_i][0]);
        }
    }
}

__global__ void KernelDivDenominatorBackward(dtype **grads,
        dtype **numerator_vals,
        dtype **denominator_vals,
        int count,
        int *dims,
        volatile dtype *block_sums,
        int *block_counters,
        dtype **denominator_grads) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    __shared__ volatile dtype square;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    int count_i = blockIdx.x;
    if (threadIdx.x == 0) {
        is_last_block = false;
        square = denominator_vals[count_i][0] * denominator_vals[count_i][0];
    }
    __syncthreads();

    int offset = blockIdx.y * blockDim.x + threadIdx.x;

    shared_sum[threadIdx.x] = offset < dims[count_i] ? grads[count_i][offset] *
        numerator_vals[count_i][offset] / square : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            DeviceAtomicAdd(denominator_grads[count_i], -shared_sum[0]);
        }
    }
}

void DivBackward(vector<dtype*> &grads, vector<dtype*> &denominator_vals,
        vector<dtype*> &numerator_vals,
        int count,
        vector<int> &dims,
        vector<dtype*> &numerator_grads,
        vector<dtype*> &denominator_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray loss_arr, denominator_val_arr, numerator_val_arr, numerator_loss_arr,
        denominator_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    denominator_val_arr.init((dtype**)denominator_vals.data(), denominator_vals.size());
    numerator_val_arr.init((dtype**)numerator_vals.data(), numerator_vals.size());
    numerator_loss_arr.init((dtype**)numerator_grads.data(), numerator_grads.size());
    denominator_loss_arr.init((dtype**)denominator_grads.data(), denominator_grads.size());
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    int block_count = DefaultBlockCount(count * max_dim);
    KernelDivNumeratorBackward<<<block_count, TPB>>>(loss_arr.value, denominator_val_arr.value,
            count, dim_arr.value, max_dim, (dtype **)numerator_loss_arr.value);
    CheckCudaError();

    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    KernelDivDenominatorBackward<<<block_dim , thread_count, thread_count * sizeof(dtype)>>>(
            loss_arr.value, numerator_val_arr.value, denominator_val_arr.value, count, dim_arr.value,
            block_sums.value, block_counters.value, (dtype **)denominator_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelFullDivForward(dtype **numerators,
        dtype **denominators,
        int count,
        int dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        results[count_i][dim_i] = numerators[count_i][dim_i] / denominators[count_i][dim_i];
    }
}

void FullDivForward(vector<dtype*> numerators,
        vector<dtype*> denominators,
        int count,
        int dim,
        vector<dtype*> &results) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray numerator_arr, denominator_arr, result_arr;
    numerator_arr.init((dtype**)numerators.data(), count);
    denominator_arr.init((dtype**)denominators.data(), count);
    result_arr.init((dtype**)results.data(), count);
    KernelFullDivForward<<<block_count, TPB>>>(numerator_arr.value, denominator_arr.value, count,
            dim, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelFullDivBackward(dtype **grads,
        dtype **numerator_vals,
        dtype **denominator_vals,
        int count,
        int dim,
        dtype **numerator_grads,
        dtype **denominator_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(numerator_grads[count_i] + dim_i, grads[count_i][dim_i] /
                denominator_vals[count_i][dim_i]);
        DeviceAtomicAdd(denominator_grads[count_i] + dim_i, -grads[count_i][dim_i] *
                numerator_vals[count_i][dim_i] /
                (denominator_vals[count_i][dim_i] * denominator_vals[count_i][dim_i]));
    }
}

void FullDivBackward(vector<dtype*> &grads,
        vector<dtype*> &denominator_vals,
        vector<dtype*> &numerator_vals,
        int count,
        int dim,
        vector<dtype*> &numerator_grads,
        vector<dtype*> &denominator_grads) {
    NumberPointerArray loss_arr, denominator_val_arr, numerator_val_arr, numerator_loss_arr,
        denominator_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    denominator_val_arr.init((dtype**)denominator_vals.data(), denominator_vals.size());
    numerator_val_arr.init((dtype**)numerator_vals.data(), numerator_vals.size());
    numerator_loss_arr.init((dtype**)numerator_grads.data(), numerator_grads.size());
    denominator_loss_arr.init((dtype**)denominator_grads.data(), denominator_grads.size());

    int block_count = DefaultBlockCount(count * dim);
    KernelFullDivBackward<<<block_count, TPB>>>(loss_arr.value, numerator_val_arr.value,
            denominator_val_arr.value, count, dim, (dtype **)numerator_loss_arr.value,
            (dtype **)denominator_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSplitForward(dtype **inputs, int *offsets, int count, int *dims, int max_dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            int offset = offsets[count_i];
            results[count_i][dim_i] = inputs[count_i][offset + dim_i];
        }
    }
}

void SplitForward(vector<dtype*> &inputs, vector<int> &offsets, int count, vector<int> &dims,
        vector<dtype*> &results) {
    NumberPointerArray input_arr, result_arr;
    input_arr.init(inputs.data(), inputs.size());
    result_arr.init(results.data(), results.size());
    IntArray offset_arr, dim_arr;
    offset_arr.init((int*)offsets.data(), offsets.size());
    dim_arr.init(dims.data(), dims.size());
    int max_dim = *max_element(dims.begin(), dims.end());

    int block_count = DefaultBlockCount(count * max_dim);
    KernelSplitForward<<<block_count, TPB>>>(input_arr.value, offset_arr.value, count,
            dim_arr.value, max_dim, result_arr.value);
    CheckCudaError();
}

__global__ void KernelSplitBackward(dtype **grads, int *offsets, int count, int *dims, int max_dim,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            int offset = offsets[count_i];
            DeviceAtomicAdd(input_grads[count_i] + offset + dim_i, grads[count_i][dim_i]);
        }
    }
}

void SplitBackward(vector<dtype*> &grads, vector<int> offsets, int count, vector<int> &dims,
        vector<dtype*> &input_grads) {
    NumberPointerArray loss_arr, input_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    input_loss_arr.init((dtype**)input_grads.data(), input_grads.size());
    IntArray offset_arr, dim_arr;
    offset_arr.init((int*)offsets.data(), offsets.size());
    dim_arr.init((int*)dims.data(), dims.size());
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelSplitBackward<<<block_count, TPB>>>(loss_arr.value, offset_arr.value, count,
            dim_arr.value, max_dim, input_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSubForward(dtype **minuend, dtype **subtrahend,
        int count,
        int *dims,
        int max_dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = minuend[count_i][dim_i] - subtrahend[count_i][dim_i];
        }
    }
}

void SubForward(vector<dtype*> &minuend,
        vector<dtype*> &subtrahend,
        int count,
        vector<int> &dims,
        vector<dtype*> &results) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray minuend_arr, subtrahend_arr, result_arr;
    minuend_arr.init((dtype**)minuend.data(), count);
    subtrahend_arr.init((dtype**)subtrahend.data(), count);
    result_arr.init((dtype**)results.data(), count);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelSubForward<<<block_count, TPB>>>((dtype**)minuend_arr.value,
            (dtype **)subtrahend_arr.value, count, dim_arr.value, max_dim,
            (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelSubBackward(dtype **grads, int count, int *dims, int max_dim,
        dtype **minuend_grads,
        dtype **subtrahend_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(minuend_grads[count_i] + dim_i, grads[count_i][dim_i]);
            DeviceAtomicAdd(subtrahend_grads[count_i] + dim_i, -grads[count_i][dim_i]);
        }
    }
}

void SubBackward(vector<dtype*> &grads, int count, vector<int> &dims,
        vector<dtype*> &minuend_grads,
        vector<dtype*> &subtrahend_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray loss_arr, minuend_loss_arr, subtrahend_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    minuend_loss_arr.init((dtype**)minuend_grads.data(), minuend_grads.size());
    subtrahend_loss_arr.init((dtype**)subtrahend_grads.data(), subtrahend_grads.size());

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelSubBackward<<<block_count, TPB>>>((dtype **)loss_arr.value, count,
            dim_arr.value, max_dim, (dtype **)minuend_loss_arr.value,
            (dtype **)subtrahend_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelPMultiBackward(dtype **grads,
        dtype **in_vals1,
        dtype **in_vals2,
        int count,
        int dim,
        dtype **in_grads1,
        dtype **in_grads2) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(in_grads1[count_i] + dim_i,
                grads[count_i][dim_i] * in_vals2[count_i][dim_i]);
        DeviceAtomicAdd(in_grads2[count_i] + dim_i,
                grads[count_i][dim_i] * in_vals1[count_i][dim_i]);
    }
}

void PMultiBackward(vector<dtype*> &grads,
        vector<dtype*> &in_vals1,
        vector<dtype*> &in_vals2,
        int count,
        int dim,
        vector<dtype*> &in_grads1,
        vector<dtype*> &in_grads2) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray grad_arr, in_vals1_arr, in_vals2_arr, in_grads1_arr,
                       in_grads2_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    in_vals1_arr.init((dtype**)in_vals1.data(), in_vals1.size());
    in_vals2_arr.init((dtype**)in_vals2.data(), in_vals2.size());
    in_grads1_arr.init((dtype**)in_grads1.data(), in_grads1.size());
    in_grads2_arr.init((dtype**)in_grads2.data(), in_grads2.size());
    KernelPMultiBackward<<<block_count, TPB>>>(grad_arr.value, in_vals1_arr.value,
            in_vals2_arr.value, count, dim, (dtype **)in_grads1_arr.value,
            (dtype **)in_grads2_arr.value);
    CheckCudaError();
}

__global__ void KernelPAddForward(dtype ***ins, int count, int dim,
        int in_count,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i+= step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype sum = ins[0][count_i][dim_i];
        for (int j = 1; j < in_count; ++j) {
            sum += ins[j][count_i][dim_i];
        }
        vals[count_i][dim_i] = sum;
    }
}

void PAddForward(vector<vector<dtype*>> &ins, int count,
        int dim,
        int in_count,
        vector<dtype*> &vals) {
    vector<shared_ptr<NumberPointerArray>> gpu_addr;
    gpu_addr.reserve(ins.size());
    for (vector<dtype*> &x : ins) {
        shared_ptr<NumberPointerArray> arr =
            make_shared<NumberPointerArray>();
        arr->init((dtype**)x.data(), x.size());
        gpu_addr.push_back(arr);
    }
    vector<dtype**> ins_gpu;
    ins_gpu.reserve(ins.size());
    for (auto &ptr : gpu_addr) {
        ins_gpu.push_back((dtype**)ptr->value);
    }

    NumberPointerPointerArray in_arr;
    in_arr.init(ins_gpu.data(), ins_gpu.size());
    NumberPointerArray out_arr;
    out_arr.init(vals.data(), vals.size());

    int block_count = DefaultBlockCount(count * dim);
    KernelPAddForward<<<block_count, TPB>>>(in_arr.value, count, dim, in_count,
            (dtype **)out_arr.value);
    CheckCudaError();
}

__global__ void KernelPAddBackward(dtype **grads, int count, int dim,
        int in_count,
        dtype ***in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int dim_mul_count = dim * count;
    for (int i = index; i < dim_mul_count * in_count; i += step) {
        int in_count_i = i / dim_mul_count;
        int dim_mul_count_i = i % dim_mul_count;
        int count_i = dim_mul_count_i / dim;
        int dim_i = dim_mul_count_i % dim;
        DeviceAtomicAdd(in_grads[in_count_i][count_i] + dim_i, grads[count_i][dim_i]);
    }
}

void PAddBackward(vector<dtype*> &grads, int count, int dim,
        int in_count,
        vector<vector<dtype*>> &in_grads) {
    vector<shared_ptr<NumberPointerArray>> gpu_addr;
    gpu_addr.reserve(in_grads.size());
    for (vector<dtype*> &x : in_grads) {
        shared_ptr<NumberPointerArray> arr =
            make_shared<NumberPointerArray>();
        arr->init((dtype**)x.data(), x.size());
        gpu_addr.push_back(arr);
    }
    vector<dtype**> in_grads_gpu;
    in_grads_gpu.reserve(in_grads.size());
    for (auto &ptr : gpu_addr) {
        in_grads_gpu.push_back((dtype **)ptr->value);
    }

    NumberPointerPointerArray in_loss_arr;
    in_loss_arr.init(in_grads_gpu.data(), in_grads_gpu.size());
    NumberPointerArray out_loss_arr;
    out_loss_arr.init((dtype**)grads.data(), grads.size());

    int block_count = DefaultBlockCount(in_count * count * dim);
    KernelPAddBackward<<<block_count, TPB>>>(out_loss_arr.value,
            count, dim, in_count, (dtype ***)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSoftMaxLoss(dtype **vals, dtype **grads,
        int *correct_count, int *answers, int batchsize, int count, int dim) {
    volatile __shared__ int opt_label;
    volatile __shared__ dtype shared_val[TPB];
    volatile __shared__ int64_t max_indexes[TPB];
    volatile __shared__ dtype scores_sum[TPB];
    volatile __shared__ dtype scores[TPB];
    int dim_i = threadIdx.x;
    int count_i = blockIdx.x;
    if (count_i == 0 && dim_i == 0) {
        *correct_count = 0;
    }
    shared_val[dim_i] = dim_i < dim ? vals[count_i][dim_i] : -1e10;
    max_indexes[dim_i] = dim_i;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (shared_val[threadIdx.x + i] > shared_val[threadIdx.x]) { // race
            shared_val[threadIdx.x] = shared_val[threadIdx.x + i]; // race
            max_indexes[threadIdx.x] = max_indexes[threadIdx.x + i]; // race
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        opt_label = max_indexes[0];
        if (answers[count_i] == opt_label) {
            atomicAdd(correct_count, 1);
        }
    }
    __syncthreads();

    dtype max_score = vals[count_i][opt_label];
    dtype score = dim_i < dim ? cuda_exp(vals[count_i][dim_i] - max_score) :
        0.0f;
    scores[dim_i] = score;
    scores_sum[dim_i] = score;

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        scores_sum[threadIdx.x] = scores_sum[threadIdx.x] +
            scores_sum[threadIdx.x + i]; // race
        __syncthreads();
    }

    if (dim_i < dim) {
        grads[count_i][dim_i] = (scores[dim_i] / scores_sum[0] -
                (dim_i == answers[count_i] ? 1 : 0)) / batchsize;
    }
}

void SoftMaxLoss(vector<dtype*> &vals, vector<dtype*> &losses, int *correct_count,
        vector<int> &answers,
        int batchsize,
        int count,
        int dim) {
    if (dim > TPB) {
        abort();
    }
    int thread_count = NextTwoIntegerPowerNumber(dim);
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size());
    IntArray answer_arr;
    answer_arr.init((int*)answers.data(), answers.size());
    KernelSoftMaxLoss<<<count, thread_count>>>(
            const_cast<dtype **>(val_arr.value),
            const_cast<dtype **>(loss_arr.value),
            correct_count,
            answer_arr.value,
            batchsize,
            count,
            dim);
    CheckCudaError();
}

__global__ void KernelCrossEntropyLoss(dtype **vals, int *answers, int count,
        dtype factor,
        dtype **losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        int answer = answers[i];
        DeviceAtomicAdd(losses[i] + answer, - 1 / vals[i][answer] * factor);
    }
}

__global__ void KernelCrossEntropgyLossValue(dtype **vals, int *answers,
        int count,
        volatile dtype *global_sum,
        int *block_counter,
        dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }
    shared_sum[threadIdx.x] = 0.0f;
    for (int i = index; i < count; i += blockDim.x * gridDim.x) {
        int answer_offset = answers[i];
        shared_sum[threadIdx.x] -= cuda_log(vals[i][answer_offset]);
    }

    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype CrossEntropyLoss(vector<dtype *> &vals, vector<int> &answers, int count,
        dtype factor,
        vector<dtype *> &losses) {
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    loss_arr.init((dtype**)losses.data(), losses.size());
    IntArray answer_arr;
    answer_arr.init((int*)answers.data(), answers.size());

    KernelCrossEntropyLoss<<<DefaultBlockCount(count), TPB>>>(val_arr.value, answer_arr.value,
            count, factor, (dtype **)loss_arr.value);
    CheckCudaError();

    int block_count = DefaultBlockCount(count);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelCrossEntropgyLossValue<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count,
            global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    return result.v * factor;
}

__global__ void KernelMultiCrossEntropyLoss(dtype **vals, int **answers,
        int count,
        int dim,
        dtype factor,
        dtype **losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype val = vals[count_i][dim_i];
        dtype grad = (answers[count_i][dim_i] ? (-1 / val) : (1 / (1 - val))) * factor;
        DeviceAtomicAdd(losses[count_i] + dim_i, grad);
    }
}

__global__ void KernelMultiCrossEntropyLossVector(dtype **in_vals,
        int **answers,
        int count,
        int dim,
        dtype **result) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype in_val = in_vals[count_i][dim_i];
        dtype v = answers[count_i][dim_i] ? -cuda_log(in_val) : -cuda_log(1 - in_val);
        result[count_i][dim_i] = v;
    }
}

template<typename T>
vector<T *> GPUArrayVectors(vector<shared_ptr<GPUArray<T>>> &ptrs, int count, int dim) {
    vector<T *> result;
    for (int i = 0; i < count; ++i) {
        shared_ptr<GPUArray<T>> e(new GPUArray<T>);
        e->init(dim);
        ptrs.push_back(e);
        result.push_back((T *)e->value);
    }
    return result;
}

dtype MultiCrossEntropyLoss(vector<dtype*> &vals, vector<vector<int>> &answers,
        int count,
        int dim,
        dtype factor,
        vector<dtype*> &losses) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), count);
    loss_arr.init((dtype**)losses.data(), count);

    vector<shared_ptr<IntArray>> answer_gpus;
    vector<int *> answer_gpu_pointers;
    for (auto &answer : answers) {
        shared_ptr<IntArray> answer_gpu(new IntArray);
        answer_gpu->init(answer.data(), answer.size());
        answer_gpus.push_back(answer_gpu);
        answer_gpu_pointers.push_back(answer_gpu->value);
    }

    IntPointerArray answer_arr;
    answer_arr.init((int**)answer_gpu_pointers.data(), count);
    KernelMultiCrossEntropyLoss<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count, dim,
            factor, (dtype **)loss_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> nums;
    vector<dtype *> logged_vec = GPUArrayVectors(nums, count, dim);

    NumberPointerArray logged_arr;
    logged_arr.init(logged_vec.data(), count);

    KernelMultiCrossEntropyLossVector<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count,
            dim, (dtype **)logged_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> ce_loss_arrs;
    vector<dtype *> ce_losses = GPUArrayVectors(ce_loss_arrs, count, 1);
    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
    }
    vector<dtype *> const_logged_arr;
    auto return_const = [](dtype *v) -> dtype* {
        return const_cast<dtype*>(v);
    };
    transform(logged_vec.begin(), logged_vec.end(), back_inserter(const_logged_arr), return_const);

    vector<int> dims;
    for (int i = 0; i < count; ++i) {
        dims.push_back(dim);
    }
    VectorSumForward(const_logged_arr, count, dims, ce_losses);

    dtype ce_loss_sum = 0.0f;

    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
        if (vec.size() != 1) {
            cerr << "vec size is not 1" << endl;
            abort();
        }
        dtype l = vec.front() * factor;
        ce_loss_sum += l;
    }

    return ce_loss_sum;
}

__global__ void KernelKLCrossEntropyLoss(dtype **vals, dtype **answers,
        int count,
        int dim,
        dtype factor,
        dtype **losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype val = vals[count_i][dim_i];
        dtype grad = -answers[count_i][dim_i] / val * factor;
        DeviceAtomicAdd(losses[count_i] + dim_i, grad);
    }
}

__global__ void KernelKLCrossEntropyLossVector(dtype **in_vals,
        dtype **answers,
        int count,
        int dim,
        dtype **result) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype in_val = in_vals[count_i][dim_i];
        dtype v = -answers[count_i][dim_i] * cuda_log(in_val);
        result[count_i][dim_i] = v;
    }
}

dtype KLCrossEntropyLoss(vector<dtype*> &vals,
        vector<shared_ptr<vector<dtype>>> &answers,
        int count,
        int dim,
        dtype factor,
        vector<dtype*> &losses) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), count);
    loss_arr.init((dtype**)losses.data(), count);

    vector<shared_ptr<NumberArray>> answer_gpus;
    vector<dtype *> answer_gpu_pointers;
    for (auto &answer : answers) {
        shared_ptr<NumberArray> answer_gpu(new NumberArray);
        answer_gpu->init(answer->data(), answer->size());
        answer_gpus.push_back(answer_gpu);
        answer_gpu_pointers.push_back(answer_gpu->value);
    }

    NumberPointerArray answer_arr;
    answer_arr.init((dtype**)answer_gpu_pointers.data(), count);
    KernelKLCrossEntropyLoss<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count, dim,
            factor, (dtype **)loss_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> nums;
    vector<dtype *> logged_vec = GPUArrayVectors(nums, count, dim);

    NumberPointerArray logged_arr;
    logged_arr.init(logged_vec.data(), count);

    KernelKLCrossEntropyLossVector<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count,
            dim, (dtype **)logged_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> ce_loss_arrs;
    vector<dtype *> ce_losses = GPUArrayVectors(ce_loss_arrs, count, 1);
    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
    }
    vector<dtype *> const_logged_arr;
    auto return_const = [](dtype *v) -> dtype* {
        return const_cast<dtype*>(v);
    };
    transform(logged_vec.begin(), logged_vec.end(), back_inserter(const_logged_arr), return_const);

    vector<int> dims;
    for (int i = 0; i < count; ++i) {
        dims.push_back(dim);
    }
    VectorSumForward(const_logged_arr, count, dims, ce_losses);

    dtype ce_loss_sum = 0.0f;

    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
        if (vec.size() != 1) {
            cerr << "vec size is not 1" << endl;
            abort();
        }
        dtype l = vec.front() * factor;
        ce_loss_sum += l;
    }

    return ce_loss_sum;
}

__global__ void KernelMax(dtype **v, int count, int dim, volatile dtype *block_maxes,
        volatile int *block_max_is,
        int *block_counters,
        int *max_indexes,
        dtype *max_vals) {
    __shared__ volatile dtype shared_max[TPB];
    __shared__ volatile int shared_max_i[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_max[threadIdx.x] = offset < dim ? v[count_i][offset] : -1e10;
    shared_max_i[threadIdx.x] = offset;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i && shared_max[threadIdx.x] < shared_max[threadIdx.x + i]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
            shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_maxes_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_maxes[block_maxes_offset] = shared_max[0];
        block_max_is[block_maxes_offset] = shared_max_i[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype max = -1e10;
        int max_i = 100000;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            if (block_maxes[offset] > max) {
                max = block_maxes[offset];
                max_i = block_max_is[offset];
            }
        }

        shared_max[threadIdx.x] = max;
        shared_max_i[threadIdx.x] = max_i;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i && shared_max[threadIdx.x + i] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
                shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_vals[count_i] = shared_max[0];
            max_indexes[count_i] = shared_max_i[0];
        }
    }
}

__global__ void KernelSingleMax(dtype **v, int count, int dim,
        int *max_indexes,
        dtype *max_vals) {
    for (int count_i = 0; count_i < count; ++count_i) {
        dtype max_val = -1e10;
        int max_i;
        for (int dim_i = 0; dim_i < dim; ++ dim_i) {
            if (v[count_i][dim_i] > max_val) {
                max_val = v[count_i][dim_i];
                max_i = dim_i;
            }
        }

        max_indexes[count_i] = max_i;
        max_vals[count_i] = max_val;
    }
}

void Max(dtype **v, int count, int dim, int *max_indexes, dtype *max_vals) {
    int thread_count = min(NextTwoIntegerPowerNumber(dim), TPB);
    int block_y_count = (dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_maxes;
    block_maxes.init(block_y_count * count);
    IntArray block_max_is, block_counters;
    block_max_is.init(block_y_count * count);
    block_counters.init(count);

    KernelMax<<<block_dim, thread_count>>>(v, count, dim, block_maxes.value, block_max_is.value,
            block_counters.value, max_indexes, max_vals);
    CheckCudaError();
#if TEST_CUDA
    NumberArray max_val_arr;
    IntArray max_indexer_arr;
    max_val_arr.init(count);
    max_indexer_arr.init(count);
    KernelSingleMax<<<1, 1>>>(v, count, dim, max_indexer_arr.value, max_val_arr.value);
    CheckCudaError();
    vector<int> max_indexer_target(count), max_indexer_gold(count);
    MyCudaMemcpy(max_indexer_target.data(), max_indexes, count * sizeof(int), cudaMemcpyDeviceToHost);
    MyCudaMemcpy(max_indexer_gold.data(), max_indexer_arr.value, count * sizeof(int),
            cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; ++i) {
        if (max_indexer_target.at(i) != max_indexer_gold.at(i)) {
            cerr << format("max_indexer_target:%1% max_indexer_gold:%2%") % max_indexer_target.at(i)
                % max_indexer_gold.at(i) << endl;
            PrintNums(v, i, dim);
            abort();
        }
    }
#endif

    CheckCudaError();
}

vector<int> Predict(vector<dtype*> &vals, int count, int dim) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    IntArray max_index_arr;
    max_index_arr.init(vals.size());
    NumberArray max_val_arr;
    max_val_arr.init(vals.size());
    Max(val_arr.value, count, dim, max_index_arr.value, max_val_arr.value);
    return max_index_arr.toCpu();
}


__global__ void KernelSum(dtype **v, int count, int dim, volatile dtype *block_sums,
        int *block_counters,
        dtype *sum_vals) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_sum[threadIdx.x] = offset < dim ? v[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            sum_vals[count_i] = shared_sum[0];
        }
    }
}

void Sum(dtype **v, int count, int dim, dtype *sum_vals) {
    int thread_count = min(NextTwoIntegerPowerNumber(dim), TPB);
    int block_y_count = (dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    KernelSum<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(v,
            count, dim, block_sums.value, block_counters.value, sum_vals);
    CheckCudaError();
}

__global__ void KernelSoftMaxLossByExp(dtype **exps, int count, int dim,
        dtype **vals,
        dtype *sums,
        dtype *max_vals,
        int *answers,
        dtype reverse_batchsize,
        dtype **grads,
        dtype *losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;

        dtype loss = exps[count_i][dim_i] / sums[count_i];
        if (dim_i == answers[count_i]) {
            loss -= 1.0f;
        }
        grads[count_i][dim_i] = loss * reverse_batchsize;
        losses[count_i] = (cuda_log(sums[count_i]) - vals[count_i][answers[count_i]] + max_vals[count_i])
            * reverse_batchsize;
    }
}

void SoftMaxLossByExp(dtype **exps, int count, int dim, dtype **vals,
        dtype *sums,
        dtype *max_vals,
        int *answers,
        dtype reverse_batchsize,
        dtype **grads,
        dtype *losses) {
    int block_count = DefaultBlockCount(dim * count);
    KernelSoftMaxLossByExp<<<block_count, TPB>>>(exps, count, dim, vals, sums, max_vals, answers,
            reverse_batchsize, (dtype **)grads, losses);
    CheckCudaError();
}

__global__ void KernelMaxScalarForward(dtype **v, int count, int* dims, int max_dim,
        volatile dtype *block_maxes,
        volatile int *block_max_is,
        int *block_counters,
        int *max_indexes,
        dtype **max_vals) {
    __shared__ volatile dtype shared_max[TPB];
    __shared__ volatile int shared_max_i[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_max[threadIdx.x] = offset < dims[count_i] ? v[count_i][offset] : -1e10;
    shared_max_i[threadIdx.x] = offset;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i && shared_max[threadIdx.x] < shared_max[threadIdx.x + i]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
            shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_maxes_offset = blockIdx.x * gridDim.y + blockIdx.y;
        int max_ii = shared_max_i[0];
        if (max_ii < 0 || max_ii >= max_dim) {
            printf("threadIdx.x == 0 after first reduce max_ii:%d v:%f\n", max_ii, shared_max[0]);
            for (int i = 0; i < TPB; ++i) {
                printf("shared_max[%d]:%f shared_max_i[%d]:%d\n", i, shared_max[i], i,
                        shared_max_i[i]);
            }
            assert(false);
        }
        block_maxes[block_maxes_offset] = shared_max[0];
        block_max_is[block_maxes_offset] = shared_max_i[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype max = -1e10;
        int max_i = 100000;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            int max_ii = block_max_is[offset];
            if (max_ii < 0 || max_ii >= max_dim) {
                printf("offset:%d is_last_block block_maxes[offset]:%f block_max_is[offset]:%d\n",
                        offset, block_maxes[offset], block_max_is[offset]);
                assert(false);
            }
            if (block_maxes[offset] > max) {
                max = block_maxes[offset];
                max_i = block_max_is[offset];
            }
        }

        shared_max[threadIdx.x] = max;
        shared_max_i[threadIdx.x] = max_i;
//        printf("max:%f max_i:%d\n", max, max_i);
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i && shared_max[threadIdx.x + i] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
                shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_vals[count_i][0] = shared_max[0];
            max_indexes[count_i] = shared_max_i[0];
            int max_ii = max_indexes[count_i];
            if (max_ii < 0 || max_ii >= max_dim) {
                printf("threadIdx.x == 0 max_i:%d count_i:%d max_val:%f\n", max_indexes[count_i],
                        count_i, max_vals[count_i][0]);
                assert(false);
            }
        }
    }
}

void MaxScalarForward(vector<dtype*> &inputs, int count, vector<int> &dims,
        vector<dtype*> &results,
        vector<int> &max_indexes) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_maxes;
    block_maxes.init(block_y_count * count);
    IntArray block_max_is, block_counters;
    block_max_is.init(block_y_count * count);
    block_counters.init(count);

    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size());
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size());
    IntArray max_index_arr;
    max_index_arr.init(max_indexes.size());

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelMaxScalarForward<<<block_dim, thread_count>>>((dtype **)input_arr.value,
            count, dim_arr.value, max_dim, block_maxes.value, block_max_is.value,
            block_counters.value,
            max_index_arr.value, (dtype **)result_arr.value);
    CheckCudaError();
    MyCudaMemcpy(max_indexes.data(), max_index_arr.value, count * sizeof(int),
            cudaMemcpyDeviceToHost);
}

__global__ void KernelMaxScalarBackward(dtype **grads, int *indexes, int count,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        DeviceAtomicAdd(input_grads[i] + indexes[i], grads[i][0]);
    }
}

void MaxScalarBackward(vector<dtype *> &grads, vector<int> &indexes, int count,
        vector<dtype*> &input_grads) {
    int block_count = DefaultBlockCount(count);
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());
    IntArray index_arr;
    index_arr.init((int*)indexes.data(), indexes.size());
    KernelMaxScalarBackward<<<block_count, TPB>>>((dtype **)grad_arr.value,
            index_arr.value, count,
            (dtype **)input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelVectorSumForward(dtype **v, int count, int *dims,
        volatile dtype *block_sums,
        int *block_counters,
        dtype **results) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_sum[threadIdx.x] = offset < dims[count_i] ? v[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            results[count_i][0] = shared_sum[0];
        }
    }
}


void VectorSumForward(vector<dtype *> &inputs, int count, vector<int> &dims,
        vector<dtype*> &results) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size());
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size());

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelVectorSumForward<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            (dtype **)input_arr.value, count, dim_arr.value, block_sums.value,
            block_counters.value, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelVectorSumBackward(dtype **grads, int count, int *dims, int max_dim,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;

        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(input_grads[count_i] + dim_i, grads[count_i][0]);
        }
    }
}

void VectorSumBackward(vector<dtype*> &grads, int count, vector<int> &dims,
        vector<dtype*> &input_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelVectorSumBackward<<<block_count, TPB>>>((dtype **)grad_arr.value, count,
            dim_arr.value, max_dim, (dtype **)input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelScaledForward(dtype **in_vals, int count, int *dims, int max_dim,
        dtype *factors,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;

        if (dim_i < dims[count_i]) {
            vals[count_i][dim_i] = in_vals[count_i][dim_i] * factors[count_i];
        }
    }
}

void ScaledForward(vector<dtype *> &in_vals, int count, vector<int> &dims, vector<dtype> &factors,
        vector<dtype *> &vals) {
    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init(in_vals.data(), in_vals.size());
    val_arr.init(vals.data(), vals.size());
    NumberArray factor_arr;
    factor_arr.init(factors.data(), count);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScaledForward<<<block_count, TPB>>>(in_val_arr.value, count, dim_arr.value, max_dim,
            factor_arr.value, val_arr.value);
}

__global__ void KernelScaledBackward(dtype **grads, int count, int *dims, int max_dim,
        dtype *factors,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;

        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i] * factors[count_i]);
        }
    }
}

void ScaledBackward(vector<dtype *> &grads, int count, vector<int> &dims, vector<dtype> &factors,
        vector<dtype *> &in_grads) {
    NumberPointerArray grad_arr, in_grad_arr;
    grad_arr.init(grads.data(), count);
    in_grad_arr.init(in_grads.data(), count);

    NumberArray factor_arr;
    factor_arr.init(factors.data(), count);

    IntArray dim_arr;
    dim_arr.init(dims.data(), count);

    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScaledBackward<<<block_count, TPB>>>(grad_arr.value, count, dim_arr.value, max_dim,
            factor_arr.value, in_grad_arr.value);
}

__global__ void KernelScalarToVectorForward(dtype* const* inputs, int count, int *dims,
        int max_dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = inputs[count_i][0];
        }
    }
}

void ScalarToVectorForward(vector<dtype*> &inputs, int count, vector<int> &dims,
        vector<dtype*> &results) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(max_dim * count);
    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size());
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), inputs.size());
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelScalarToVectorForward<<<block_count, TPB>>>((dtype* *)input_arr.value,
            count, dim_arr.value, max_dim, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarToVectorBackward(dtype **grads, int count, int *dims,
        volatile dtype *block_sums,
        int *block_counters,
        dtype **input_grads) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_sum[threadIdx.x] = offset < dims[count_i] ? grads[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            DeviceAtomicAdd(input_grads[count_i], shared_sum[0]);
        }
    }
}

void ScalarToVectorBackward(vector<dtype*> &grads, int count, vector<int> &dims,
        vector<dtype*> &input_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());

    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    NumberPointerArray grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    NumberPointerArray input_grad_arr;
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelScalarToVectorBackward<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            (dtype **)grad_arr.value, count, dim_arr.value, block_sums.value, block_counters.value,
            (dtype **)input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelBiasForward(dtype **in_vals, dtype *bias, int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        vals[count_i][dim_i] = in_vals[count_i][dim_i] + bias[dim_i];
    }
}

void BiasForward(vector<dtype*> &in_vals, dtype *bias, int count, int dim,
        vector<dtype *> &vals) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray in_arr, val_arr;
    in_arr.init(in_vals.data(), in_vals.size());
    val_arr.init(vals.data(), vals.size());
    KernelBiasForward<<<block_count, TPB>>>(in_arr.value, bias, count, dim,
            (dtype **)val_arr.value);
}

__global__ void KernelBiasBackward(dtype **grads, int count, int dim,
        dtype *bias_grads,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(bias_grads + dim_i, grads[count_i][dim_i]);
        DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i]);
    }
}

void BiasBackward(vector<dtype *> &grads, int count, int dim, dtype *bias_grad,
        vector<dtype *> input_grads) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init(grads.data(), grads.size());
    input_grad_arr.init(input_grads.data(), input_grads.size());
    KernelBiasBackward<<<block_count, TPB>>>(grad_arr.value, count, dim, bias_grad,
            (dtype **)input_grad_arr.value);
}

__global__ void KernelSquareSum(dtype *v, int len, volatile dtype *global_sum,
        int *block_counter, dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }
    shared_sum[threadIdx.x] = 0.0f;
    for (int i = index; i < len; i += blockDim.x * gridDim.x) {
        shared_sum[threadIdx.x] += v[i] * v[i];
    }

    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype SquareSum(dtype *v, int len) {
    int block_count = DefaultBlockCount(len);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB>>>(v, len,
            global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    return result.v;
}

__global__ void KernelSquareSum(dtype *v, bool *indexers,
        int count,
        int dim,
        volatile dtype *global_sum,
        int *block_counter,
        dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = 0.0f;
        is_last_block = false;
    }
    int count_i = index / dim;
    if (index < count * dim && indexers[count_i]) {
        shared_sum[threadIdx.x] = v[index] * v[index];
    } else {
        shared_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype SquareSum(dtype *v, bool *indexers, int count, int dim) {
    int block_count = DefaultBlockCountWithoutLimit(count * dim);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB>>>(v, indexers,
            count, dim, global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    return result.v;
}

__global__ void KernelRescale(dtype *v, int len, dtype scale) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        v[i] *= scale;
    }
}

void Rescale(dtype *v, int len, dtype scale) {
    int block_count = DefaultBlockCount(len);
    KernelRescale<<<block_count, TPB>>>(v, len, scale);
    CheckCudaError();
}

__global__ void KernelUpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias,
        dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype x) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        if (!is_bias) {
            grad[i] += val[i] * reg;
        }
        aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
        aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
            grad[i];
        dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) * x;
        dtype square_plus_eps = aux_square[i] + eps;
        val[i] = val[i] - aux_mean[i] * lr_t / cuda_sqrt(square_plus_eps);
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdam<<<block_count, TPB>>>(val, grad, row, col, is_bias, aux_mean,
            aux_square,
            iter,
            belta1,
            belta2,
            alpha,
            reg,
            eps,
            x);
    CheckCudaError();
}

__global__ void KernelUpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias,
        dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype x) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
        aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
            grad[i];
        dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) * x;
        dtype square_plus_eps = aux_square[i] + eps;
        val[i] = (1 - (is_bias? 0.0f : reg)) * val[i] - aux_mean[i] * lr_t /
            cuda_sqrt(square_plus_eps);
    }
}

void UpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdamW<<<block_count, TPB>>>(val, grad, row, col, is_bias, aux_mean,
            aux_square,
            iter,
            belta1,
            belta2,
            alpha,
            reg,
            eps,
            x);
    CheckCudaError();
}

__global__ void KernelUpdateAdam(dtype *val, dtype *grad, int row, int col,
        dtype *aux_mean,
        dtype *aux_square,
        bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        int count_i = i / row;
        if (indexers[count_i]) {
            if (row > 1 && col > 1) {
                grad[i] += val[i] * reg;
            }
            aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
            aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
                grad[i];
            dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2,
                        iters[count_i] + 1)) / (1 - cuda_pow(belta1,
                            iters[count_i] + 1));
            dtype square_plus_eps = aux_square[i] + eps;
            val[i] = val[i] - aux_mean[i] * lr_t / cuda_sqrt(square_plus_eps);
        }
    }
}

__global__ void KernelSelfPlusIters(bool *indexers, int *iters,
        int count) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        if (indexers[i]) {
            ++iters[i];
        }
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdam<<<block_count, TPB>>>(val, grad, row, col, aux_mean,
            aux_square, indexers, iters, belta1, belta2, alpha, reg, eps);
    CheckCudaError();
    block_count = DefaultBlockCount(col);
    KernelSelfPlusIters<<<block_count, TPB>>>(indexers, iters, col);
    CheckCudaError();
}

__global__ void KernelUpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        if (row > 1 && col > 1) {
            grad[i] += val[i] * reg;
        }
        aux_square[i] = aux_square[i] + grad[i] * grad[i];
        val[i] = val[i] - grad[i] * alpha / cuda_sqrt(aux_square[i] + eps);
    }
}

void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdagrad<<<block_count, TPB>>>(val, grad, row, col, aux_square,
            alpha, reg, eps);
    CheckCudaError();
}

__global__ void KernelUpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        int count_i = i / col;
        if (indexers[count_i]) {
            if (row > 1 && col > 1) {
                grad[i] += val[i] * reg;
            }
            aux_square[i] = aux_square[i] + grad[i] * grad[i];
            val[i] = val[i] - grad[i] * alpha / cuda_sqrt(aux_square[i] + eps);
        }
    }
}

void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdagrad<<<block_count, TPB>>>(val, grad, row, col, aux_square,
            indexers, alpha, reg, eps);
    CheckCudaError();
}

void *GraphHostAlloc() {
    void *m;
    CallCuda(cudaHostAlloc(&m, 10000000, cudaHostAllocWriteCombined));
    if (m == NULL) {
        abort();
    }
    return m;
}

}
