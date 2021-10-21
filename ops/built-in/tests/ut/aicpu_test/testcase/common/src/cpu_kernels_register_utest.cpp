#include "gtest/gtest.h"

#include <limits.h>
#include "cce/fwk_adpt_struct.h"
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "cce/fwk_adpt_struct.h"
#include "cce/aicpu_engine_struct.h"
#include "aicpu_context.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "async_cpu_kernel.h"

using namespace std;
using namespace aicpu;

namespace {
    const char* Test  = "Test";
    const char* TestAsync = "TestAsync";
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

class TestAsyncKernel : public AsyncCpuKernel {
public:
    ~TestAsyncKernel() = default;
    typedef std::function<void(uint32_t status)> DoneCallBack;
    uint32_t ComputeAsync(CpuKernelContext &ctx, DoneCallBack done) override
    {
        cout << "TestAsyncKernel success." << endl;
        uint32_t status = KERNEL_STATUS_OK;
        done(status);
        return KERNEL_STATUS_OK;
    }
};
REGISTER_CPU_KERNEL(TestAsync, TestAsyncKernel);

} // namespace aicpu

class CPU_KERNEL_REGISTAR_UTest : public testing::Test {};

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
    totalLen = 4 * FWKAdapter::kExtInfoHeadSize + sizeof(int32_t) + sizeof(uint64_t) + 2 * sizeof(FWKAdapter::ShapeAndType);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    //init ext info 0: shapeType 
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    //init ext info 1: bitmap
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_BITMAP;
    extInfo->infoLen = sizeof(uint64_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    uint64_t *bitMap = reinterpret_cast<uint64_t *>(extInfoBuf + extInfoOffset);
    *bitMap = 0;
    extInfoOffset += extInfo->infoLen;
    // init ext info 2: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;
    // init ext info 3: output ShapeAndType
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *outputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    outputs->dims[0]=8;
    outputs->dims[1]=2;
    outputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    free(parambase);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernel_Notiling) {
  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Test", "Test")
      .Input({"x", DT_INT32, {1}, nullptr})
      .Output({"y", DT_INT32, {1}, nullptr})
      .Attr("attr", true);

  std::string nodeDefStr;
  nodeDef->SerializeToString(nodeDefStr);
  std::vector<void*> io_addrs;
  uint32_t input[2000] = {0};
  uint32_t output[2000] = {0};
  io_addrs.push_back(reinterpret_cast<void*>(input));
  io_addrs.push_back(reinterpret_cast<void*>(output));
  uint32_t param_len = static_cast<uint32_t>(sizeof(AicpuParamHead) + sizeof(uint32_t) + nodeDefStr.length());
  param_len += 2 * static_cast<uint32_t>(sizeof(uint64_t));

  std::vector<char> taskExtInfo;
  uint64_t totalLen = 0;
  totalLen = 6 * FWKAdapter::kExtInfoHeadSize + sizeof(int32_t) + sizeof(uint64_t) +
             2 * sizeof(FWKAdapter::ShapeAndType) + 2 * sizeof(uint32_t);
  taskExtInfo.resize(totalLen, 0);
  uint32_t extInfoOffset = 0;
  char* extInfoBuf = taskExtInfo.data();
  // init ext info 0: shapeType
  FWKAdapter::ExtInfo* extInfo = reinterpret_cast<FWKAdapter::ExtInfo*>(extInfoBuf);
  extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
  extInfo->infoLen = sizeof(int32_t);
  extInfoOffset += FWKAdapter::kExtInfoHeadSize;
  int32_t* shapeType = reinterpret_cast<int32_t*>(extInfoBuf + extInfoOffset);
  *shapeType = 3;
  extInfoOffset += extInfo->infoLen;
  // init ext info 1: bitmap
  extInfo = reinterpret_cast<FWKAdapter::ExtInfo*>(extInfoBuf + extInfoOffset);
  extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_BITMAP;
  extInfo->infoLen = sizeof(uint64_t);
  extInfoOffset += FWKAdapter::kExtInfoHeadSize;
  uint64_t* bitMap = reinterpret_cast<uint64_t*>(extInfoBuf + extInfoOffset);
  *bitMap = 0;
  extInfoOffset += extInfo->infoLen;
  // init ext info 2: input ShapeAndType 	359
  extInfo = reinterpret_cast<FWKAdapter::ExtInfo*>(extInfoBuf + extInfoOffset);
  extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
  extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
  extInfoOffset += FWKAdapter::kExtInfoHeadSize;
  FWKAdapter::ShapeAndType* inputs = reinterpret_cast<FWKAdapter::ShapeAndType*>(extInfoBuf + extInfoOffset);
  inputs->dims[0] = 2;
  inputs->dims[1] = 4;
  inputs->dims[2] = LLONG_MIN;
  extInfoOffset += extInfo->infoLen;
  // init ext info 3: output ShapeAndType
  extInfo = reinterpret_cast<FWKAdapter::ExtInfo*>(extInfoBuf + extInfoOffset);
  extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
  extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
  extInfoOffset += FWKAdapter::kExtInfoHeadSize;
  FWKAdapter::ShapeAndType* outputs = reinterpret_cast<FWKAdapter::ShapeAndType*>(extInfoBuf + extInfoOffset);
  outputs->dims[0] = 8;
  outputs->dims[1] = 2;
  outputs->dims[2] = LLONG_MIN;
  extInfoOffset += extInfo->infoLen;

  // init ext info 4: unknown shape input index
  extInfo = reinterpret_cast<FWKAdapter::ExtInfo*>(extInfoBuf + extInfoOffset);
  extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_UNKNOWN_SHAPE_INPUT_INDEX;
  extInfo->infoLen = sizeof(uint32_t);
  extInfoOffset += FWKAdapter::kExtInfoHeadSize;
  uint32_t* input_index_ptr = reinterpret_cast<uint32_t*>(extInfoBuf + extInfoOffset);
  input_index_ptr[0] = 0;
  extInfoOffset += extInfo->infoLen;

  // init ext info 4: unknown shape output index
  extInfo = reinterpret_cast<FWKAdapter::ExtInfo*>(extInfoBuf + extInfoOffset);
  extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_UNKNOWN_SHAPE_OUTPUT_INDEX;
  extInfo->infoLen = sizeof(uint32_t);
  extInfoOffset += FWKAdapter::kExtInfoHeadSize;
  uint32_t* output_index_ptr = reinterpret_cast<uint32_t*>(extInfoBuf + extInfoOffset);
  output_index_ptr[0] = 0;
  extInfoOffset += extInfo->infoLen;

  AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
  char* parambase = (char*)malloc(param_len * sizeof(char));
  char* param = parambase;

  memcpy(parambase, reinterpret_cast<const char*>(&paramHead), sizeof(AicpuParamHead));
  param += sizeof(AicpuParamHead);
  memcpy(param, reinterpret_cast<const char*>(io_addrs.data()),
         io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
  param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
  uint32_t nodeDefLen = nodeDefStr.length();
  memcpy(param, reinterpret_cast<const char*>(&nodeDefLen), sizeof(uint32_t));
  param += sizeof(uint32_t);
  memcpy(param, static_cast<const char*>(nodeDefStr.data()), nodeDefStr.length());

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
    EXPECT_NE(ret, KERNEL_STATUS_OK);
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
    EXPECT_NE(ret, KERNEL_STATUS_OK);
    free(parambase);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernelWithSessionInfo)
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
    totalLen = 5 * FWKAdapter::kExtInfoHeadSize + sizeof(int32_t) + sizeof(uint64_t) + 2 * sizeof(FWKAdapter::ShapeAndType) + sizeof(SessionInfo);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    //init ext info 0: shapeType 
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    //init ext info 1: bitmap
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_BITMAP;
    extInfo->infoLen = sizeof(uint64_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    uint64_t *bitMap = reinterpret_cast<uint64_t *>(extInfoBuf + extInfoOffset);
    *bitMap = 0;
    extInfoOffset += extInfo->infoLen;
    // init ext info 2: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;
    // init ext info 3: output ShapeAndType
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *outputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    outputs->dims[0]=8;
    outputs->dims[1]=2;
    outputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;

    // init ext info 4: SessionInfo
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
    extInfo->infoLen = sizeof(SessionInfo);
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    SessionInfo *sessionInfo = reinterpret_cast<SessionInfo *>(extInfoBuf + extInfoOffset);
    sessionInfo->sessionId = 0;
    sessionInfo->kernelId = 0;
    sessionInfo->sessFlag = false;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    free(parambase);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernelWithSessionInfoAsyncKernel_success)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "TestAsync", "TestAsync")
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
    totalLen = 6 * FWKAdapter::kExtInfoHeadSize + sizeof(int32_t) +
               sizeof(uint64_t) + 2 * sizeof(FWKAdapter::ShapeAndType) +
               sizeof(SessionInfo) + sizeof(aicpu::FWKAdapter::AsyncWait);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    //init ext info 0: shapeType 
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    //init ext info 1: bitmap
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_BITMAP;
    extInfo->infoLen = sizeof(uint64_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    uint64_t *bitMap = reinterpret_cast<uint64_t *>(extInfoBuf + extInfoOffset);
    *bitMap = 0;
    extInfoOffset += extInfo->infoLen;
    // init ext info 2: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;
    // init ext info 3: output ShapeAndType
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *outputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    outputs->dims[0]=8;
    outputs->dims[1]=2;
    outputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;

    // init ext info 4: SessionInfo
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
    extInfo->infoLen = sizeof(SessionInfo);
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    SessionInfo *sessionInfo = reinterpret_cast<SessionInfo *>(extInfoBuf + extInfoOffset);
    sessionInfo->sessionId = 1;
    sessionInfo->kernelId = 0;
    sessionInfo->sessFlag = true;
    extInfoOffset += extInfo->infoLen;

    // init ext info 5: AsyncWait
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
    extInfo->infoLen = sizeof(aicpu::FWKAdapter::AsyncWait);
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    aicpu::FWKAdapter::AsyncWait *asyncWait = reinterpret_cast<aicpu::FWKAdapter::AsyncWait *>(extInfoBuf + extInfoOffset);
    asyncWait->waitType = FWKAdapter::FWK_ADPT_WAIT_TYPE_EVENT;
    asyncWait->waitId = 0;
    asyncWait->timeOut = 0;
    asyncWait->reserved = 0;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    free(parambase);
}


TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernelWithSessionInfoAsyncKernel_failed)
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
    totalLen = 6 * FWKAdapter::kExtInfoHeadSize + sizeof(int32_t) +
               sizeof(uint64_t) + 2 * sizeof(FWKAdapter::ShapeAndType) +
               sizeof(SessionInfo) + sizeof(aicpu::FWKAdapter::AsyncWait);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    //init ext info 0: shapeType 
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    //init ext info 1: bitmap
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_BITMAP;
    extInfo->infoLen = sizeof(uint64_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    uint64_t *bitMap = reinterpret_cast<uint64_t *>(extInfoBuf + extInfoOffset);
    *bitMap = 0;
    extInfoOffset += extInfo->infoLen;
    // init ext info 2: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;
    // init ext info 3: output ShapeAndType
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *outputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    outputs->dims[0]=8;
    outputs->dims[1]=2;
    outputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;

    // init ext info 4: SessionInfo
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
    extInfo->infoLen = sizeof(SessionInfo);
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    SessionInfo *sessionInfo = reinterpret_cast<SessionInfo *>(extInfoBuf + extInfoOffset);
    sessionInfo->sessionId = 2;
    sessionInfo->kernelId = 0;
    sessionInfo->sessFlag = true;
    extInfoOffset += extInfo->infoLen;

    // init ext info 5: AsyncWait
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
    extInfo->infoLen = sizeof(aicpu::FWKAdapter::AsyncWait);
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    aicpu::FWKAdapter::AsyncWait *asyncWait = reinterpret_cast<aicpu::FWKAdapter::AsyncWait *>(extInfoBuf + extInfoOffset);
    asyncWait->waitType = FWKAdapter::FWK_ADPT_WAIT_TYPE_EVENT;
    asyncWait->waitId = 0;
    asyncWait->timeOut = 0;
    asyncWait->reserved = 0;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
    free(parambase);
}

TEST_F(CPU_KERNEL_REGISTAR_UTest, DeviceRunKernelWithSessionInfoAsyncKernel_param_error)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "TestAsync", "TestAsync")
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
    totalLen = 6 * FWKAdapter::kExtInfoHeadSize + sizeof(int32_t) +
               sizeof(uint64_t) + 2 * sizeof(FWKAdapter::ShapeAndType) +
               sizeof(SessionInfo) + sizeof(aicpu::FWKAdapter::AsyncWait);
    taskExtInfo.resize(totalLen, 0);
    uint32_t extInfoOffset = 0;
    char *extInfoBuf = taskExtInfo.data();
    //init ext info 0: shapeType 
    FWKAdapter::ExtInfo *extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE;
    extInfo->infoLen = sizeof(int32_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    int32_t *shapeType = reinterpret_cast<int32_t *>(extInfoBuf + extInfoOffset);
    *shapeType = 3;
    extInfoOffset += extInfo->infoLen;
    //init ext info 1: bitmap
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_BITMAP;
    extInfo->infoLen = sizeof(uint64_t);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    uint64_t *bitMap = reinterpret_cast<uint64_t *>(extInfoBuf + extInfoOffset);
    *bitMap = 0;
    extInfoOffset += extInfo->infoLen;
    // init ext info 2: input ShapeAndType 	359
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *inputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    inputs->dims[0]=2;
    inputs->dims[1]=4;
    inputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;
    // init ext info 3: output ShapeAndType
    extInfo = reinterpret_cast<FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE;
    extInfo->infoLen = sizeof(FWKAdapter::ShapeAndType);
    extInfoOffset += FWKAdapter::kExtInfoHeadSize;
    FWKAdapter::ShapeAndType *outputs = reinterpret_cast<FWKAdapter::ShapeAndType *>(extInfoBuf + extInfoOffset);
    outputs->dims[0]=8;
    outputs->dims[1]=2;
    outputs->dims[2]=LLONG_MIN;
    extInfoOffset += extInfo->infoLen;

    // init ext info 4: SessionInfo
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO;
    extInfo->infoLen = sizeof(SessionInfo);
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    SessionInfo *sessionInfo = reinterpret_cast<SessionInfo *>(extInfoBuf + extInfoOffset);
    sessionInfo->sessionId = 3;
    sessionInfo->kernelId = 0;
    sessionInfo->sessFlag = true;
    extInfoOffset += extInfo->infoLen;

    // init ext info 5: AsyncWait
    extInfo = reinterpret_cast<aicpu::FWKAdapter::ExtInfo *>(extInfoBuf + extInfoOffset);
    extInfo->infoType = aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT;
    // error asyncwait len
    extInfo->infoLen = sizeof(aicpu::FWKAdapter::AsyncWait) + 10;
    extInfoOffset += aicpu::FWKAdapter::kExtInfoHeadSize;
    aicpu::FWKAdapter::AsyncWait *asyncWait = reinterpret_cast<aicpu::FWKAdapter::AsyncWait *>(extInfoBuf + extInfoOffset);
    asyncWait->waitType = FWKAdapter::FWK_ADPT_WAIT_TYPE_EVENT;
    asyncWait->waitId = 0;
    asyncWait->timeOut = 0;
    asyncWait->reserved = 0;

    AicpuParamHead paramHead = {param_len, 2, totalLen, reinterpret_cast<uintptr_t>(taskExtInfo.data())};
    char* parambase = (char*)malloc(param_len * sizeof(char));
    char* param = parambase;

    memcpy(parambase, reinterpret_cast<const char *>(&paramHead), sizeof(AicpuParamHead));
    param += sizeof(AicpuParamHead);
    memcpy(param, reinterpret_cast<const char *>(io_addrs.data()), io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t)));
    param += io_addrs.size() * static_cast<uint32_t>(sizeof(uint64_t));
    uint32_t nodeDefLen = nodeDefStr.length();
    memcpy(param, reinterpret_cast<const char *>(&nodeDefLen), sizeof(uint32_t));
    param += sizeof(uint32_t);
    memcpy(param, static_cast<const char *>(nodeDefStr.data()), nodeDefStr.length());

    uint32_t ret = RunCpuKernel(parambase);
    EXPECT_NE(ret, KERNEL_STATUS_OK);
    free(parambase);
}

