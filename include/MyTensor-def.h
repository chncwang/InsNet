#ifndef BasicTensor_DEF
#define BasicTensor_DEF

#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include "Def.h"
#include "serializable.h"
#include <iostream>
#include <iostream>

using namespace Eigen;

namespace n3ldg_cpu {

struct Tensor1D : public N3LDGSerializable {
    dtype *v;
    int dim;

    Tensor1D();

    virtual ~Tensor1D();

    virtual void init(int ndim);

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

    virtual Json::Value toJson() const;

    virtual void fromJson(const Json::Value &json);

    virtual void print() const;
};

struct Tensor2D : public N3LDGSerializable {
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


    dtype* operator[](const int icol);

    const dtype* operator[](const int icol) const;

    virtual void assignAll(dtype a);

    Tensor2D& operator=(const std::vector<dtype> &a);

    Tensor2D& operator=(const std::vector<std::vector<dtype> > &a);

    Tensor2D& operator=(const Tensor2D &a);

    void random(dtype bound);

    void norm2one(dtype norm = 1.0);

    virtual Json::Value toJson() const;

    virtual void fromJson(const Json::Value &json);
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

    Tensor1D();
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&);
    void init(int len) override;
    void initOnMemoryAndDevice(int len);
    ~Tensor1D();

    virtual std::string name() const;

    Tensor1D& operator=(const Tensor1D &tensor);

    Tensor1D& operator=(dtype v);

    void print() const override;

    void random(dtype bound) override;

    bool verify(const char *message) const;

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;

private:
    void initOnDevice(int len);
};

struct Tensor2D : public n3ldg_cpu::Tensor2D, public Transferable {
    dtype *value = NULL;

    Tensor2D();
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&);
    ~Tensor2D();

    void init(int row, int col) override;

    virtual std::string name() const;

    void print() const override;

    void zero();

    void random(dtype bound);

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

#endif
