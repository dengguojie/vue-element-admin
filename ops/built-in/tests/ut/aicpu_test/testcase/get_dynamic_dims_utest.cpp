#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "securec.h"

using namespace std;
using namespace aicpu;

class GET_DYNAMIC_DIMS_UT : public testing::Test {};

TEST_F(GET_DYNAMIC_DIMS_UT, INT32_Success)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int32_t> x1{ 3, 2, 4, 1 };
  vector<int32_t> x2{ 1, 2, 1 };
  vector<int32_t> x3{ 16, 112, 112, 3, 4 };
  vector<int32_t> dims(3);
  vector<int64_t> shape_info_attr{
      4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4 };

  NodeDefBuilder(node_def.get(), "GetDynamicDims", "GetDynamicDims")
      .Input({"x1", DT_INT32, {4}, x1.data()})
      .Input({"x2", DT_INT32, {3}, x2.data()})
      .Input({"x3", DT_INT32, {5}, x3.data()})
      .Output({"dims", DT_INT32, {3}, dims.data()})
      .Attr("N", 3)
      .Attr("shape_info", shape_info_attr);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<int32_t> expectDims{4, 112, 112};
  EXPECT_EQ(dims, expectDims);
}

TEST_F(GET_DYNAMIC_DIMS_UT, INT64_Success)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int64_t> x1{ 3, 2, 4, 1 };
  vector<int64_t> x2{ 1, 2, 1 };
  vector<int64_t> x3{ 16, 112, 112, 3, 4 };
  vector<int64_t> dims(3);
  vector<int64_t> shape_info_attr{
      4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4 };

  NodeDefBuilder(node_def.get(), "GetDynamicDims", "GetDynamicDims")
      .Input({"x1", DT_INT64, {4}, x1.data()})
      .Input({"x2", DT_INT64, {3}, x2.data()})
      .Input({"x3", DT_INT64, {5}, x3.data()})
      .Output({"dims", DT_INT64, {3}, dims.data()})
      .Attr("N", 3)
      .Attr("shape_info", shape_info_attr);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<int64_t> expectDims{4, 112, 112};
  EXPECT_EQ(dims, expectDims);
}

TEST_F(GET_DYNAMIC_DIMS_UT, OutputNum_Fail)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int32_t> x1{ 3, 2, 4, 1 };
  vector<int32_t> x2{ 1, 2, 1 };
  vector<int32_t> x3{ 16, 112, 112, 3, 4 };
  vector<int32_t> dims(3);
  vector<int64_t> shape_info_attr{
      4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4 };

  NodeDefBuilder(node_def.get(), "GetDynamicDims", "GetDynamicDims")
      .Input({"x1", DT_INT32, {4}, x1.data()})
      .Input({"x2", DT_INT32, {3}, x2.data()})
      .Input({"x3", DT_INT32, {5}, x3.data()})
      .Output({"dims", DT_INT32, {3}, dims.data()})
      .Output({"dims2", DT_INT64, {3}, dims.data()})
      .Attr("N", 3)
      .Attr("shape_info", shape_info_attr);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
