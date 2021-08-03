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
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_GEMM_UT : public testing::Test {};

#define CREATE_NODEDEF1(shapes, data_types, datas)                 \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "GEMM", "GEMM")                   \
      .Input({"a", data_types[0], shapes[0], datas[0]})            \
      .Input({"b", data_types[1], shapes[1], datas[1]})            \
      .Input({"c", data_types[2], shapes[2], datas[2]})            \
      .Input({"alpha", data_types[3], shapes[3], datas[3]})        \
      .Input({"beta", data_types[4], shapes[4], datas[4]})         \
      .Output({"y", data_types[5], shapes[5], datas[5]})           \
      .Attr("transpose_a", false)                                  \
      .Attr("transpose_b", false)

#define CREATE_NODEDEF2(shapes, data_types, datas)                 \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "GEMM", "GEMM")                   \
      .Input({"a", data_types[0], shapes[0], datas[0]})            \
      .Input({"b", data_types[1], shapes[1], datas[1]})            \
      .Input({"c", data_types[2], shapes[2], datas[2]})            \
      .Input({"alpha", data_types[3], shapes[3], datas[3]})        \
      .Input({"beta", data_types[4], shapes[4], datas[4]})         \
      .Output({"y", data_types[5], shapes[5], datas[5]})           \
      .Attr("transpose_a", true)                                   \
      .Attr("transpose_b", false)

TEST_F(TEST_GEMM_UT, Not_Transpose_a_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                 DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 2}, {}, {}, {2, 2}};
  float input0[6] = {1, 2, 3, 4, 5, 6};
  float input1[6] = {7, 8, 9, 10, 11, 12};
  float input2[4] = {1, -1, -1, 1};
  float input3[1] = {1};
  float input4[1] = {-1};
  float output[4] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2,
                          (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF1(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[4] = {57, 65, 140, 153};
  bool compare = CompareResult(output, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_GEMM_UT, Transpose_a_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                 DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 2}, {3, 2}, {2, 2}, {}, {}, {2, 2}};
  float input0[6] = {1, 4, 2, 5, 3, 6};
  float input1[6] = {7, 8, 9, 10, 11, 12};
  float input2[4] = {1, -1, -1, 1};
  float input3[1] = {1};
  float input4[1] = {-1};
  float output[4] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2,
                          (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF2(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[4] = {57, 65, 140, 153};
  bool compare = CompareResult(output, output_exp, 4);
  EXPECT_EQ(compare, true);
}
