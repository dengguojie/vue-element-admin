#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include <mockcpp/ChainingMockHelper.h>

using namespace std;
using namespace aicpu;


class TEST_Meshgrid_UTest : public testing::Test {
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

TEST_F(TEST_Meshgrid_UTest, MeshgridKernel_xy_Success)
{
    // raw data
    int input0[4] = {1, 2, 3, 1};
    int input1[3] = {4, 5, 6};
    int input2[6] = {7, 8, 9, 4, 2, 2};
    int output1[72] = {0};
    int output2[72] = {0};
    int output3[72] = {0};

    int expected1[72] = {1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1};
    int expected2[72] = {4, 4, 4, 4, 4, 4,
                         4, 4, 4, 4, 4, 4,
                         4, 4, 4, 4, 4, 4,
                         4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5,
                         5, 5, 5, 5, 5, 5,
                         5, 5, 5, 5, 5, 5,
                         5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6,
                         6, 6, 6, 6, 6, 6,
                         6, 6, 6, 6, 6, 6,
                         6, 6, 6, 6, 6, 6};
    int expected3[72] = {7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Meshgrid");

    // set attr
    auto indexing = CpuKernelUtils::CreateAttrValue();
    indexing->SetString("xy");
    nodeDef->AddAttrs("indexing", indexing.get());

    //set input
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {4};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_INT32);
    inputTensor0->SetData(input0);
    inputTensor0->SetDataSize(4 * sizeof(int));

    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {3};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_INT32);
    inputTensor1->SetData(input1);
    inputTensor1->SetDataSize(3 * sizeof(int));

    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {6};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(input2);
    inputTensor2->SetDataSize(6 * sizeof(int));

    //set output
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    outputTensor1->SetDataType(DT_INT32);
    outputTensor1->SetData(output1);
    outputTensor1->SetDataSize(72 * sizeof(int));

    auto outputTensor2 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor2, nullptr);
    outputTensor2->SetDataType(DT_INT32);
    outputTensor2->SetData(output2);
    outputTensor2->SetDataSize(72 * sizeof(int));

    auto outputTensor3 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor3, nullptr);
    outputTensor3->SetDataType(DT_INT32);
    outputTensor3->SetData(output3);
    outputTensor3->SetDataSize(72 * sizeof(int));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(output1, expected1, sizeof(output1)));
    EXPECT_EQ(0, std::memcmp(output2, expected2, sizeof(output2)));
    EXPECT_EQ(0, std::memcmp(output3, expected3, sizeof(output3)));
}

TEST_F(TEST_Meshgrid_UTest, MeshgridKernel_ij_Success)
{
    // raw data
    int input0[4] = {1, 2, 3, 1};
    int input1[3] = {4, 5, 6};
    int input2[6] = {7, 8, 9, 4, 2, 2};
    int output1[72] = {0};
    int output2[72] = {0};
    int output3[72] = {0};

    int expected1[72] = {1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 2, 2,
                         2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3,
                         3, 3, 3, 3, 3, 3,
                         3, 3, 3, 3, 3, 3,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1};
    int expected2[72] = {4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6,
                         4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6,
                         4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6,
                         4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6};
    int expected3[72] = {7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2,
                         7, 8, 9, 4, 2, 2};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("Meshgrid");

    // set attr
    auto indexing = CpuKernelUtils::CreateAttrValue();
    indexing->SetString("ij");
    nodeDef->AddAttrs("indexing", indexing.get());

    //set input
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {4};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_INT32);
    inputTensor0->SetData(input0);
    inputTensor0->SetDataSize(4 * sizeof(int));

    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {3};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_INT32);
    inputTensor1->SetData(input1);
    inputTensor1->SetDataSize(3 * sizeof(int));

    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {6};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(input2);
    inputTensor2->SetDataSize(6 * sizeof(int));

    //set output
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    outputTensor1->SetDataType(DT_INT32);
    outputTensor1->SetData(output1);
    outputTensor1->SetDataSize(72 * sizeof(int));

    auto outputTensor2 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor2, nullptr);
    outputTensor2->SetDataType(DT_INT32);
    outputTensor2->SetData(output2);
    outputTensor2->SetDataSize(72 * sizeof(int));

    auto outputTensor3 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor3, nullptr);
    outputTensor3->SetDataType(DT_INT32);
    outputTensor3->SetData(output3);
    outputTensor3->SetDataSize(72 * sizeof(int));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(output1, expected1, sizeof(output1)));
    EXPECT_EQ(0, std::memcmp(output2, expected2, sizeof(output2)));
    EXPECT_EQ(0, std::memcmp(output3, expected3, sizeof(output3)));
}