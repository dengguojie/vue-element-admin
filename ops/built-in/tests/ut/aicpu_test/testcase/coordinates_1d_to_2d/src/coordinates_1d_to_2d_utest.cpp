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

class TEST_COORDINATES_1D_TO_2D_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "Coordinates1DTo2D", "Coordinates1DTo2D") \
      .Input({"x", data_types[0], shapes[0], datas[0]})                    \
      .Input({"shape", data_types[1], shapes[1], datas[1]})                \
      .Output({"row", data_types[2], shapes[2], datas[2]})                 \
      .Output({"col", data_types[3], shapes[3], datas[3]})                 \
      .Output({"n", data_types[4], shapes[4], datas[4]})    

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_0_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32,
                                 DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
  int32_t input0[1] = {0};
  int32_t input1[4] = {1, 1, 4, 4};
  int32_t output[3] = {-1, -1, -1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)&output[0],
                          (void *)&output[1], (void *)&output[2]};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[3] = {0, 0, 4};
  bool compare = CompareResult(output, output_exp, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_5_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32,
                                 DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
  int32_t input0[1] = {5};
  int32_t input1[4] = {1, 1, 4, 4};
  int32_t output[3] = {-1, -1, -1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)&output[0],
                          (void *)&output[1], (void *)&output[2]};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[3] = {1, 1, 4};
  bool compare = CompareResult(output, output_exp, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_15_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32,
                                 DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
  int32_t input0[1] = {15};
  int32_t input1[4] = {1, 1, 4, 4};
  int32_t output[3] = {-1, -1, -1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)&output[0],
                          (void *)&output[1], (void *)&output[2]};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[3] = {3, 3, 4};
  bool compare = CompareResult(output, output_exp, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_0_int64_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64,
                                 DT_INT64};
  vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
  int64_t input0[1] = {0};
  int64_t input1[4] = {1, 1, 4, 4};
  int64_t output[3] = {-1, -1, -1};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)&output[0],
                          (void *)&output[1], (void *)&output[2]};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int64_t output_exp[3] = {0, 0, 4};
  bool compare = CompareResult(output, output_exp, 3);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COORDINATES_1D_TO_2D_UT, ThreadId_0_uint64_SUCCESS) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64, DT_UINT64,
                                 DT_UINT64};
  vector<vector<int64_t>> shapes = {{}, {4}, {}, {}, {}};
  uint64_t input0[1] = {0};
  uint64_t input1[4] = {1, 1, 4, 4};
  uint64_t output[3] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)&output[0],
                          (void *)&output[1], (void *)&output[2]};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[3] = {0, 0, 4};
  bool compare = CompareResult(output, output_exp, 3);
  EXPECT_EQ(compare, true);
}
