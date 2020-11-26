#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_types.h"
#include "node_def_builder.h"
#include "cpu_kernel_utils.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace aicpu;

class TEST_LOGGING_KERNEL_UTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInt8) {
  cout<<"Test int8 assert kernel begin."<<endl;
  bool input_condition = false;
  int8_t input_data[4] = {1, 2, 3, 4};

  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_INT8, {4}, (void *)input_data})
      .Attr("summarize", 3);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInt8Dim2) {
  cout<<"Test Int8Dim2 assert kernel begin."<<endl;
  bool input_condition = false;
  int8_t input_data[4] = {1, 2, 3, 4};

  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_INT8, {2, 2}, (void *)input_data})
      .Attr("summarize", 4);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInt8Dim3) {
  cout<<"Test Int8Dim3 assert kernel begin."<<endl;
  bool input_condition = false;
  int8_t input_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_INT8, {1, 3, 4}, (void *)input_data})
      .Attr("summarize", 20);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInt8Dim4) {
  cout<<"Test Int8Dim4 assert kernel begin."<<endl;
  bool input_condition = false;
  int8_t input_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_INT8, {2, 2, 3}, (void *)input_data})
      .Attr("summarize", 20);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, Assertfloat) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  float input_data[4] = {1.0, 2.1, 3.0, 4.7};

  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT, {4}, (void *)input_data})
      .Attr("summarize", 4);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, Assertfloat16) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  std::vector<Eigen::half> input_data;
  for(int i = 0; i < 4; i++){
    input_data.push_back(static_cast<Eigen::half>(i));
  }
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT16, {4}, (void *)input_data.data()})
      .Attr("summarize", 4);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, Assertfloat16Dim2) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  std::vector<Eigen::half> input_data;
  for(int i = 0; i < 4; i++){
    input_data.push_back(static_cast<Eigen::half>(i));
  }
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT16, {4, 1}, (void *)input_data.data()})
      .Attr("summarize", 4);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, Assertfloat16Dim3) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  std::vector<Eigen::half> input_data;
  for(int i = 0; i < 8; i++){
    input_data.push_back(static_cast<Eigen::half>(i));
  }
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT16, {2, 2, 2}, (void *)input_data.data()})
      .Attr("summarize", 10);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, Assertfloat16Dim4) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  std::vector<Eigen::half> input_data;
  for(int i = 0; i < 16; i++){
    input_data.push_back(static_cast<Eigen::half>(i));
  }
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT16, {2, 2, 1, 4}, (void *)input_data.data()})
      .Attr("summarize", 20);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInputConditionTrue) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = true;
  std::vector<Eigen::half> input_data;
  for(int i = 0; i < 16; i++){
    input_data.push_back(static_cast<Eigen::half>(i));
  }
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT16, {2, 2, 1, 4}, (void *)input_data.data()})
      .Attr("summarize", 20);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInputConditionNoScalar) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = true;
  std::vector<Eigen::half> input_data;
  for(int i = 0; i < 16; i++){
    input_data.push_back(static_cast<Eigen::half>(i));
  }
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {1}, (void *)&input_condition})
      .Input({"input_data", DT_FLOAT16, {2, 2, 1, 4}, (void *)input_data.data()})
      .Attr("summarize", 20);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_PARAM_INVALID);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInputData2) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  int32_t input_data1[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  int8_t input_data2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  double input_data3[8] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};
  int16_t input_data4[8] = {100, 200, 300, 400, 500, 600, 700, 800};
  uint64_t input_data5[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  bool input_data6[8] = {true, false, true, false, false, true, false, true};
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_INT32, {2, 4}, (void *)input_data1})
      .Input({"input_data", DT_INT8, {2, 2, 2}, (void *)input_data2})
      .Input({"input_data", DT_DOUBLE, {8, 1}, (void *)input_data3})
      .Input({"input_data", DT_UINT32, {8, 1}, (void *)input_data1})
      .Input({"input_data", DT_UINT8, {8, 1}, (void *)input_data2})
      .Input({"input_data", DT_INT16, {4, 2}, (void *)input_data4})
      .Input({"input_data", DT_UINT16, {4, 2}, (void *)input_data4})
      .Input({"input_data", DT_UINT64, {4, 2}, (void *)input_data5})
      .Input({"input_data", DT_INT64, {1, 8}, (void *)input_data5})
      .Input({"input_data", DT_BOOL, {1, 8}, (void *)input_data6})
      .Attr("summarize", 20);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInputString) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  std::string input_data[8] = {"Test", "logging", "kernel", "utest", "assert", "input", "string"};
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_STRING, {8}, (void *)input_data})
      .Attr("summarize", 6);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_LOGGING_KERNEL_UTest, AssertInputNotSupport) {
  cout<<"Test float assert kernel begin."<<endl;
  bool input_condition = false;
  std::string input_data[8] = {"Assert", "Input", "Not", "Support"};
  auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(nodeDef.get(), "Assert", "Assert")
      .Input({"input_condition", DT_BOOL, {}, (void *)&input_condition})
      .Input({"input_data", DT_DUAL, {2, 4}, (void *)input_data})
      .Attr("summarize", 6);

  CpuKernelContext ctx(HOST);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

  cout<<"Test assert kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}