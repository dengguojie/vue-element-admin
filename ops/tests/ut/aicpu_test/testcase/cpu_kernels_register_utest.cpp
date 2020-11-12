#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include <limits.h>
#include "cce/fwk_adpt_struct.h"
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

using namespace std;
using namespace aicpu;

namespace {
    const char* Test  = "Test";
}

namespace aicpu {
class TestKernel : public CpuKernel {
public:
    ~TestKernel() = default;

protected:
    uint32_t Compute(CpuKernelContext &ctx) override
    {
        cout << "TestKernel success." << endl;
        return KERNEL_STATUS_OK;
    }
};
REGISTER_CPU_KERNEL(Test, TestKernel);
} // namespace aicpu

class CPU_KERNEL_REGISTAR_UTest : public testing::Test {
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

TEST_F(CPU_KERNEL_REGISTAR_UTest, GetAndRunKernel)
{
    auto typeVec = CpuKernelRegister::Instance().GetAllRegisteredOpTypes();
    EXPECT_NE(typeVec.size(), 0);
    auto testKernel = CpuKernelRegister::Instance().GetCpuKernel("Test");
    EXPECT_NE(testKernel, nullptr);

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Test", "Test");
    CpuKernelContext ctx(HOST);
    ctx.Init(nodeDef.get());
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    testKernel = CpuKernelRegister::Instance().GetCpuKernel("Test1");
    EXPECT_EQ(testKernel, nullptr);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernel)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Test", "Test")
        .Input({"x", DT_INT32, {1}, nullptr})
        .Output({"y", DT_INT32, {1}, nullptr})
        .Attr("attr", true);

    std::string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    std::vector<void *> io_addrs;
    uint32_t input[5] = {0};
    uint32_t output[5] = {0};
    io_addrs.push_back(reinterpret_cast<void*>(input));
    io_addrs.push_back(reinterpret_cast<void*>(output));
    uint32_t param_len = static_cast<uint32_t>(sizeof(AicpuParamHead) + sizeof(uint32_t) + nodeDefStr.length());
    param_len += 2 * static_cast<uint32_t>(sizeof(uint64_t));

    std::vector<char> taskExtInfo;
    uint64_t totalLen = 0;
    totalLen = 3 * FWKAdapter::kExtInfoHeadSize + 4 + 2 * sizeof(FWKAdapter::ShapeAndType);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    // init ext info 1: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;
    // init ext info 2: output ShapeAndType
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *outputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    outputs->dims[0]=8;
    outputs->dims[1]=2;
    outputs->dims[2]=LLONG_MIN;
    extInfoOffset += sizeof(FWKAdapter::ShapeAndType);

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size()* static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    free(parambase);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernel_ShapeTypeFailed)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Test", "Test")
        .Input({"x", DT_INT32, {1}, nullptr})
        .Output({"y", DT_INT32, {1}, nullptr})
        .Attr("attr", true);

    std::string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    std::vector<void *> io_addrs;
    uint32_t input[5] = {0};
    uint32_t output[5] = {0};
    io_addrs.push_back(reinterpret_cast<void*>(input));
    io_addrs.push_back(reinterpret_cast<void*>(output));
    uint32_t param_len = static_cast<uint32_t>(sizeof(AicpuParamHead) + sizeof(uint32_t) + nodeDefStr.length());
    param_len += 2 * static_cast<uint32_t>(sizeof(uint64_t));

    std::vector<char> taskExtInfo;
    uint64_t totalLen = 0;
    totalLen = 1 * FWKAdapter::kExtInfoHeadSize + 4;
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = 0;  // wrong len
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size()* static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
    free(parambase);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernel_InputShapeFailed)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Test", "Test")
        .Input({"x", DT_INT32, {1}, nullptr})
        .Output({"y", DT_INT32, {1}, nullptr})
        .Attr("attr", true);

    std::string nodeDefStr;
    nodeDef->SerializeToString(nodeDefStr);
    std::vector<void *> io_addrs;
    uint32_t input[5] = {0};
    uint32_t output[5] = {0};
    io_addrs.push_back(reinterpret_cast<void*>(input));
    io_addrs.push_back(reinterpret_cast<void*>(output));
    uint32_t param_len = static_cast<uint32_t>(sizeof(AicpuParamHead) + sizeof(uint32_t) + nodeDefStr.length());
    param_len += 2 * static_cast<uint32_t>(sizeof(uint64_t));

    std::vector<char> taskExtInfo;
    uint64_t totalLen = 0;
    totalLen = 2 * FWKAdapter::kExtInfoHeadSize + 4 + 1 * sizeof(FWKAdapter::ShapeAndType);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    // init ext info 1: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = 2 * sizeof(FWKAdapter::ShapeAndType); // wrong len
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size()* static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
    free(parambase);
}
