#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "node_def_builder.h"
#include "cpu_kernel_utils.h"
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

template <typename T>
bool CompareResult(T output[], T expectOutput[], int num) {
    bool result = true;
    for (int i = 0; i < num; ++i) {
        if (output[i] != expectOutput[i]) {
            cout << "output[" << i << "] = ";
            cout << output[i];
            cout << "expectOutput[" << i << "] =";
            cout << expectOutput[i];
            result = false;
        }
    }
    return result;
}

class TEST_CONCATV2_ST : public testing::Test {
protected:
    virtual void SetUp() {}

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(TEST_CONCATV2_ST, TestConcatV2_host_01)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    float input[22] = {22, 32.3, -78.0, -28.5, 77, 99, 77, 89, 22, 32.3, -78.0,
        -28.5, 77, 99, 77, 45.7, 89.5, 90, 2, 1, 22, 32.3};
    float output[44] = {0};
    int32_t concat_dim = 0;
    int n=2;
    NodeDefBuilder(nodeDef.get(), "ConcatV2", "ConcatV2")
        .Input({ "x0", DT_FLOAT, { 2, 11 }, (void *)input })
        .Input({ "x1", DT_FLOAT, { 2, 11 }, (void *)input })
        .Input({ "concat_dim", DT_INT32, {}, (void *)&concat_dim })
        .Output({ "y", DT_FLOAT, { 4, 11 }, (void *)output })
        .Attr("N", n);
    string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    cout << "ConcatV2 nodeDef: " << nodeDefStr << endl;
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    float expectOut[44] = {22.0, 32.3, -78.0, -28.5, 77.0, 99.0, 77.0, 89.0, 22.0,
        32.3, -78.0, -28.5, 77.0, 99.0, 77.0, 45.7, 89.5, 90.0, 2.0, 1.0, 22.0,
        32.3, 22.0, 32.3, -78.0, -28.5, 77.0, 99.0, 77.0, 89.0, 22.0, 32.3,
        -78.0, -28.5, 77.0, 99.0, 77.0, 45.7, 89.5, 90.0, 2.0, 1.0, 22.0, 32.3};
    EXPECT_EQ(CompareResult<float>(output, expectOut, 44), true);
    for (int i = 0; i < 10; ++i) {
        cout << output[i] << ",";
    }
    cout << endl;
}

TEST_F(TEST_CONCATV2_ST, TestConcatV2_host_02)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    int64_t input[4] = {1,2,3,4};
    int64_t output[8] = {0};
    int32_t concat_dim = 0;
    int n=2;
    NodeDefBuilder(nodeDef.get(), "ConcatV2", "ConcatV2")
        .Input({ "x0", DT_INT64, { 2, 2 }, (void *)input })
        .Input({ "x1", DT_INT64, { 2, 2 }, (void *)input })
        .Input({ "concat_dim", DT_INT32, {}, (void *)&concat_dim })
        .Output({ "y", DT_INT64, { 4, 2 }, (void *)output })
        .Attr("N", n);
    string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    cout << "ConcatV2 nodeDef: " << nodeDefStr << endl;
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    int64_t expectOut[8] = {1, 2, 3, 4, 1, 2, 3, 4};
    EXPECT_EQ(CompareResult<int64_t>(output, expectOut, 8), true);
    for (int i = 0; i < 8; ++i) {
        cout << output[i] << ",";
    }
    cout << endl;
}

TEST_F(TEST_CONCATV2_ST, TestConcatV2_host_03)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    int32_t input[4] = {1,2,3,4};
    int32_t output[8] = {0};
    int32_t concat_dim = 0;
    int n=2;
    NodeDefBuilder(nodeDef.get(), "ConcatV2", "ConcatV2")
        .Input({ "x0", DT_INT32, { 2, 2 }, (void *)input })
        .Input({ "x1", DT_INT32, { 2, 2 }, (void *)input })
        .Input({ "concat_dim", DT_INT32, {}, (void *)&concat_dim })
        .Output({ "y", DT_INT32, { 4, 2 }, (void *)output })
        .Attr("N", n);
    string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    cout << "ConcatV2 nodeDef: " << nodeDefStr << endl;
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    int32_t expectOut[8] = {1, 2, 3, 4, 1, 2, 3, 4};
    EXPECT_EQ(CompareResult<int32_t>(output, expectOut, 8), true);
    for (int i = 0; i < 8; ++i) {
        cout << output[i] << ",";
    }
    cout << endl;
}

TEST_F(TEST_CONCATV2_ST, TestConcatV2_host_04)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    int8_t input[4] = {1,2,3,4};
    int8_t output[8] = {0};
    int32_t concat_dim = 0;
    int n=2;
    NodeDefBuilder(nodeDef.get(), "ConcatV2", "ConcatV2")
        .Input({ "x0", DT_INT8, { 2, 2 }, (void *)input })
        .Input({ "x1", DT_INT8, { 2, 2 }, (void *)input })
        .Input({ "concat_dim", DT_INT32, {}, (void *)&concat_dim })
        .Output({ "y", DT_INT8, { 4, 2 }, (void *)output })
        .Attr("N", n);
    string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    cout << "ConcatV2 nodeDef: " << nodeDefStr << endl;
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    int8_t expectOut[8] = {1, 2, 3, 4, 1, 2, 3, 4};
    EXPECT_EQ(CompareResult<int8_t>(output, expectOut, 8), true);
    for (int i = 0; i < 8; ++i) {
        cout << output[i] << ",";
    }
    cout << endl;
}

TEST_F(TEST_CONCATV2_ST, TestConcatV2_host_05)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    Eigen::half input[4];
    for (int i = 1; i < 5; ++i) {
        input[i-1] = (Eigen::half)i;
    }
    Eigen::half output[8];
    for (int i = 1; i < 9; ++i) {
        output[i-1] = (Eigen::half)i;
    }
    int32_t concat_dim = 0;
    int n=2;
    NodeDefBuilder(nodeDef.get(), "ConcatV2", "ConcatV2")
        .Input({ "x0", DT_FLOAT16, { 2, 2 }, (void *)input })
        .Input({ "x1", DT_FLOAT16, { 2, 2 }, (void *)input })
        .Input({ "concat_dim", DT_INT32, {}, (void *)&concat_dim })
        .Output({ "y", DT_FLOAT16, { 4, 2 }, (void *)output })
        .Attr("N", n);
    string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    cout << "ConcatV2 nodeDef: " << nodeDefStr << endl;
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    Eigen::half expectOut[8];
    for (int i = 1; i < 5; ++i) {
        expectOut[i-1] = (Eigen::half)i;
    }
    for (int i = 5; i < 9; ++i) {
        expectOut[i-1] = (Eigen::half)(i - 4);
    }
    EXPECT_EQ(CompareResult<Eigen::half>(output, expectOut, 8), true);
    for (int i = 0; i < 8; ++i) {
        cout << output[i] << ",";
    }
    cout << endl;
}


TEST_F(TEST_CONCATV2_ST, TestConcatV2_device)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    float input[22] = {22, 32.3, -78.0, -28.5, 77, 99, 77, 89, 22, 32.3, -78.0,
        -28.5, 77, 99, 77, 45.7, 89.5, 90, 2, 1, 22, 32.3};
    float output[44] = {0};
    int32_t concat_dim = 0;
    int n=2;
    NodeDefBuilder(nodeDef.get(), "ConcatV2", "ConcatV2")
        .Input({ "x0", DT_FLOAT, { 2, 11 }, (void *)input })
        .Input({ "x1", DT_FLOAT, { 2, 11 }, (void *)input })
        .Input({ "concat_dim", DT_INT32, {}, (void *)&concat_dim })
        .Output({ "y", DT_FLOAT, { 4, 11 }, (void *)output })
        .Attr("N", n);
    string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    cout << "ConcatV2 nodeDef: " << nodeDefStr << endl;
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    float expectOut[44] = {22.0, 32.3, -78.0, -28.5, 77.0, 99.0, 77.0, 89.0, 22.0,
        32.3, -78.0, -28.5, 77.0, 99.0, 77.0, 45.7, 89.5, 90.0, 2.0, 1.0, 22.0,
        32.3, 22.0, 32.3, -78.0, -28.5, 77.0, 99.0, 77.0, 89.0, 22.0, 32.3,
        -78.0, -28.5, 77.0, 99.0, 77.0, 45.7, 89.5, 90.0, 2.0, 1.0, 22.0, 32.3};
    EXPECT_EQ(CompareResult<float>(output, expectOut, 44), true);
}

