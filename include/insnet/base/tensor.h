#ifndef INSNET_TENSOR_H
#define INSNET_TENSOR_H

#include "transferable.h"
#include "insnet/base/eigen-def.h"
#include "insnet/base/memory.h"
#include "cereal/cereal.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

namespace insnet {

namespace cpu {

struct Tensor1D {
    dtype *v = nullptr;
    int dim = 0;
    int ref_count_ = 1;
    std::shared_ptr<MemoryContainer> memory_container_ = nullptr;

    Tensor1D() = default;

    virtual ~Tensor1D();

    virtual void init(int dim);

    virtual void init(int dim, const std::shared_ptr<MemoryContainer> &container);

    virtual bool isInitialized() const {
        return v != nullptr;
    }

    void retain();

    void release();

    virtual void releaseMemory();

    void zero();

    std::string toString() const;

    const Mat mat() const;

    Mat mat();

    const Mat tmat() const;

    Mat tmat();

    const Vec vec() const;

    Vec vec();

    dtype& operator[](const int i);

    const dtype& operator[](const int i) const;

    Tensor1D& operator=(const dtype &a);

    Tensor1D& operator=(const std::vector<dtype> &a);

    Tensor1D& operator=(const Tensor1D &a);

    virtual void random(dtype bound);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(dim);
        ar(cereal::binary_data(v, dim * sizeof(dtype)));
    }

    virtual void print() const;

    virtual std::vector<dtype> toCpu() const;

    virtual void checkIsNumber() const;
};

struct Tensor2D {
    dtype *v;
    int col, row, size;

    Tensor2D();

    virtual ~Tensor2D();

    virtual void init(int nrow, int ncol);

    virtual void print() const;

    std::string toString() const;

    void zero();

    const Mat mat() const;

    Mat mat();

    const Vec vec() const;

    Vec vec();

    virtual bool isInitialized() const {
        return v != nullptr;
    }

    dtype* operator[](const int icol);

    const dtype* operator[](const int icol) const;

    virtual void assignAll(dtype a);

    Tensor2D& operator=(const std::vector<dtype> &a);

    Tensor2D& operator=(const std::vector<std::vector<dtype> > &a);

    Tensor2D& operator=(const Tensor2D &a);

    virtual void random(dtype bound);

    virtual void randomNorm(dtype sd);

    void norm2one(dtype norm = 1.0);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(row);
        ar(col);
        ar(cereal::binary_data(v, row * col * sizeof(dtype)));
    }
};

}

#if USE_GPU

namespace cuda {

using insnet::dtype;

bool Verify(dtype *host, dtype* device, int len, const char* message);

struct Tensor1D : public insnet::cpu::Tensor1D, public Transferable {
    dtype *value = nullptr;

    Tensor1D();
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&);
    void init(int len) override;
    void init(int dim, const std::shared_ptr<MemoryContainer> &container) override;
    virtual bool isInitialized() const override {
        return value != nullptr;
    }
    void initOnMemoryAndDevice(int len);
    void initOnMemory(int len);
    void releaseMemory() override;
    ~Tensor1D();

    virtual std::string name() const;

    Tensor1D& operator=(const Tensor1D &tensor);

    Tensor1D& operator=(dtype v);

    void print() const override;

    virtual std::vector<dtype> toCpu() const override;

    void random(dtype bound) override;

    bool verify(const char *message) const;

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
    void checkIsNumber() const override;

private:
    void initOnDevice(int len);
};

struct Tensor2D : public insnet::cpu::Tensor2D, public Transferable {
    dtype *value = nullptr;

    Tensor2D();
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&);
    ~Tensor2D();

    void init(int row, int col) override;

    virtual bool isInitialized() const override {
        return value != nullptr;
    }

    virtual std::string name() const;

    void print() const override;

    void zero();

    void random(dtype bound) override;

    void randomNorm(dtype sd) override;

    bool verify(const char* message);

    void assignAll(dtype a) override;

    void initOnMemoryAndDevice(int row, int col);

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
private:
    void initOnDevice(int row, int col);
};


}
#endif

void initAndZeroTensors(std::vector<cpu::Tensor1D *> &tensors, const std::vector<int> &dims,
        const std::vector<std::string> &signatures);

}

#endif
