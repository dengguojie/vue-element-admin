#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include <algorithm>
#include <Eigen/Core>
#include <stdlib.h>

using namespace std;
using namespace aicpu;


class TEST_TOPK_STest : public testing::Test {
protected:
    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

//TEST_DT_FLOAT16
TEST_F(TEST_TOPK_STest, TEST_DT_FLOAT16)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    Eigen::half input[24];
    for (int i = 0; i < 24; i++) {
        input[i] = Eigen::half(rand() % 50);
    }

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_FLOAT16);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(Eigen::half));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    Eigen::half output[ind] = {Eigen::half(0)};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_FLOAT16);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(Eigen::half));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	Eigen::half *p1=(Eigen::half *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<Eigen::half>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_FLOAT
TEST_F(TEST_TOPK_STest, TEST_DT_FLOAT)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    float input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_FLOAT);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(float));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    float output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_FLOAT);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(float));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	float *p1=(float *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<float>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_DOUBLE
TEST_F(TEST_TOPK_STest, TEST_DT_DOUBLE)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    double input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_DOUBLE);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(double));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    double output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_DOUBLE);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(double));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	double *p1=(double *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<double>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_UINT8
TEST_F(TEST_TOPK_STest, TEST_DT_UINT8)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    uint8_t input[24] = {96, 97, 15, 5,
                           4,  3,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 45, 90,
                           23, 33, 56, 91,
                           25, 20, 2, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_UINT8);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(uint8_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    uint8_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_UINT8);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint8_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	uint8_t *p1=(uint8_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<uint8_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_INT8
TEST_F(TEST_TOPK_STest, TEST_DT_INT8)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    int8_t input[24] = {96, 97, 15, 5,
                           4,  3,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 45, 90,
                           23, 33, 56, 91,
                           25, 20, 2, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_INT8);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(int8_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    int8_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT8);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(int8_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	int8_t *p1=(int8_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<int8_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_UINT16
TEST_F(TEST_TOPK_STest, TEST_DT_UINT16)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    uint16_t input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_UINT16);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(uint16_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    uint16_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_UINT16);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint16_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	uint16_t *p1=(uint16_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<uint16_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_INT16
TEST_F(TEST_TOPK_STest, TEST_DT_INT16)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    int16_t input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_INT16);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(int16_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    int16_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT16);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(int16_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	int16_t *p1=(int16_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<int16_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_UINT32
TEST_F(TEST_TOPK_STest, TEST_DT_UINT32)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    uint32_t input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_UINT32);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(uint32_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    uint32_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_UINT32);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint32_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	uint32_t *p1=(uint32_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<uint32_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_INT32
TEST_F(TEST_TOPK_STest, TEST_DT_INT32)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    int32_t input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_INT32);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(int32_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    int32_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT32);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(int32_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	int32_t *p1=(int32_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<int32_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_UINT64
TEST_F(TEST_TOPK_STest, TEST_DT_UINT64)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    uint64_t input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_UINT64);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(uint64_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    uint64_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_UINT64);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(uint64_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	uint64_t *p1=(uint64_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<uint64_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}
//TEST_DT_INT64
TEST_F(TEST_TOPK_STest, TEST_DT_INT64)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("TopK");
    //set x
    int64_t input[24] = {96, 97, 15, 5,
                           4,  161,  2, 3,
                           4, 67, 8, 9,
                           35, 21, 100, 90,
                           23, 201, 56, 91,
                           25, 20, 300, 15,
                           };

    auto inputTensor = nodeDef->AddInputs();
    EXPECT_NE(inputTensor, nullptr);


    auto aicpuShape = inputTensor->GetTensorShape();
    std::vector<int64_t> shapes = {24};
    aicpuShape->SetDimSizes(shapes);

    inputTensor->SetDataType(DT_INT64);
    inputTensor->SetData(input);
    inputTensor->SetDataSize(24 * sizeof(int64_t));

    //set k
    int32_t k = 7;
    auto kTensor = nodeDef->AddInputs();
    EXPECT_NE(kTensor, nullptr);
    kTensor->SetDataType(DT_INT32);
    kTensor->SetData(&k);

    const int32_t ind = k;
    int64_t output[ind] = {0};
    int32_t indices[ind] = {0};

    //set output
    auto valueTensor = nodeDef->AddOutputs();
    EXPECT_NE(valueTensor, nullptr);
    valueTensor->SetDataType(DT_INT64);
    valueTensor->SetData(output);
    valueTensor->SetDataSize(ind * sizeof(int64_t));

    auto indicesTensor = nodeDef->AddOutputs();
    EXPECT_NE(indicesTensor, nullptr);
    indicesTensor->SetDataType(DT_INT32);
    indicesTensor->SetData(indices);
    indicesTensor->SetDataSize(ind * sizeof(int32_t));

    auto sorted = CpuKernelUtils::CreateAttrValue();
    sorted->SetBool(true);
    nodeDef->AddAttrs("sorted", sorted.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    
    auto valueAddr=valueTensor->GetData();
	int64_t *p1=(int64_t *)valueAddr;
    auto indicesAddr=indicesTensor->GetData();
	int32_t *p2=(int32_t *)indicesAddr;
    cout << endl;
    cout << "topk_output:" << endl;
    for (int i = 0; i < k; i++) {
        cout << p1[i] << "\t";
    }
    cout << endl;
    for (int i = 0; i < k; i++) {
        cout << p2[i] << "\t";
    }
    cout << endl;
    sort(input, input + 24, greater<int64_t>());
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(input[i], p1[i]);
    }
}