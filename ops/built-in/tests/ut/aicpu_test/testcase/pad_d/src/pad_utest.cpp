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
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_PADD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, paddings)                        \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "PadD", "PadD")                                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})                            \
      .Output({"y", data_types[1], shapes[1], datas[1]})                           \
      .Attr("paddings", paddings);

TEST_F(TEST_PADD_UT, DATA_SUCCESS1) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}};
  vector<vector<int64_t>> paddings = {{1, 1}, {1, 1}};
  int32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, paddings);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[24] = {0, 0, 0, 0, 0, 0,
                            0, 1, 2, 3, 4, 0,
                            0, 5, 6, 7, 8, 0,
                            0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PADD_UT, DATA_SUCCESS2) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}};
  vector<vector<int64_t>> paddings = {{1, 1}, {1, 1}};
  float input[8] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};
  float output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, paddings);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[24] = {0, 0, 0, 0, 0, 0,
                            0, 1.1, 2.1, 3.1, 4.1, 0,
                            0, 5.1, 6.1, 7.1, 8.1, 0,
                            0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PADD_UT, DATA_SUCCESS3) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}};
  vector<vector<int64_t>> paddings = {};
  int32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t output[8] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, paddings);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  bool compare = CompareResult(output, output_exp, 8);
  EXPECT_EQ(compare, true);
}