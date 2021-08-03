#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_CASE_CONDITION_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "CaseCondition", "CaseCondition")         \
      .Input({"x", data_types[0], shapes[0], datas[0]})                    \
      .Output({"y", data_types[1], shapes[1], datas[1]})                   \
      .Attr("algorithm", std::string("LU"))

TEST_F(TEST_CASE_CONDITION_UT, OUT_0_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {1,0,1};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {0};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_0_2_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {2,0,1};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {0};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_1_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {1,1,1};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {1};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_2_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {2,1,1};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {2};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_3_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {1,2,1};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {3};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_4_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {2,2,1};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {4};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_5_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int32_t input0[3] = {2,2,3};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {5};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_5_int64_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  int64_t input0[3] = {2,2,3};
  int32_t output[1] = {-1};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {5};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_5_uint64_SUCCESS) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  uint64_t input0[3] = {2,2,3};
  int32_t output[1] = {0};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[1] = {5};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CASE_CONDITION_UT, OUT_5_uint32_Failed) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3}, {}};
  uint32_t input0[3] = {2,2,3};
  uint32_t output[1] = {0};
  vector<void *> datas = {(void *)input0, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}
