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

class TEST_PAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "Pad", "Pad")                                     \
    .Input({"x", data_types[0], shapes[0], datas[0]})                              \
    .Input({"paddings", data_types[2], shapes[2], datas[2]})                       \
    .Output({"y", data_types[1], shapes[1], datas[1]});                            \

TEST_F(TEST_PAD_UT, DATA_SUCCESS1) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int64_t pad[4] = {1, 1, 1, 1};
  int32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[24] = {0, 0, 0, 0, 0, 0,
                            0, 1, 2, 3, 4, 0,
                            0, 5, 6, 7, 8, 0,
                            0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS2) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int64_t pad[4] = {1, 1, 1, 1};
  float input[8] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};
  float output[24] = {0.0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[24] = {0, 0, 0, 0, 0, 0,
                            0, 1.1, 2.1, 3.1, 4.1, 0,
                            0, 5.1, 6.1, 7.1, 8.1, 0,
                            0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS3) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int64_t pad[4] = {1, 1, 1, 1};
  Eigen::half input[8] = {(Eigen::half)1.1, (Eigen::half)2.1, (Eigen::half)3.1, (Eigen::half)4.1,
                          (Eigen::half)5.1, (Eigen::half)6.1, (Eigen::half)7.1, (Eigen::half)8.1};
  Eigen::half output[24] = {(Eigen::half)0.0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  Eigen::half output_exp[24] = {(Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0,
                                (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)1.1,
                                (Eigen::half)2.1, (Eigen::half)3.1, (Eigen::half)4.1, (Eigen::half)0.0,
                                (Eigen::half)0.0, (Eigen::half)5.1, (Eigen::half)6.1, (Eigen::half)7.1,
                                (Eigen::half)8.1, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0,
                                (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS4) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {}};
  int64_t pad[0];
  int32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[24] = {1, 2, 3, 4, 5, 6, 7, 8};
  bool compare = CompareResult(output, output_exp, 8);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS5) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int32_t pad[4] = {1, 1, 1, 1};
  int32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[24] = {0, 0, 0, 0, 0, 0,
                            0, 1, 2, 3, 4, 0,
                            0, 5, 6, 7, 8, 0,
                            0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS6) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int32_t pad[4] = {1, 1, 1, 1};
  float input[8] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};
  float output[24] = {0.0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[24] = {0, 0, 0, 0, 0, 0,
                          0, 1.1, 2.1, 3.1, 4.1, 0,
                          0, 5.1, 6.1, 7.1, 8.1, 0,
                          0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS7) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int32_t pad[4] = {1, 1, 1, 1};
  Eigen::half input[8] = {(Eigen::half)1.1, (Eigen::half)2.1, (Eigen::half)3.1, (Eigen::half)4.1,
                          (Eigen::half)5.1, (Eigen::half)6.1, (Eigen::half)7.1, (Eigen::half)8.1};
  Eigen::half output[24] = {(Eigen::half)0.0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  Eigen::half output_exp[24] = {(Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0,
                                (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)1.1,
                                (Eigen::half)2.1, (Eigen::half)3.1, (Eigen::half)4.1, (Eigen::half)0.0,
                                (Eigen::half)0.0, (Eigen::half)5.1, (Eigen::half)6.1, (Eigen::half)7.1,
                                (Eigen::half)8.1, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0,
                                (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0, (Eigen::half)0.0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS8) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {}};
  int32_t pad[0];
  int32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t output[8] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t output_exp[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  bool compare = CompareResult(output, output_exp, 8);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS9) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int64_t pad[4] = {1, 1, 1, 1};
  int8_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int8_t output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int8_t output_exp[24] = {0, 0, 0, 0, 0, 0,
                           0, 1, 2, 3, 4, 0,
                           0, 5, 6, 7, 8, 0,
                           0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS10) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int64_t pad[4] = {1, 1, 1, 1};
  double input[8] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};
  double output[24] = {0.0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  double output_exp[24] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 1.1, 2.1, 3.1, 4.1, 0.0,
                           0.0, 5.1, 6.1, 7.1, 8.1, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS11) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {}};
  uint64_t pad[0];
  uint8_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint8_t output[8] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint8_t output_exp[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  bool compare = CompareResult(output, output_exp, 8);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS12) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int32_t pad[4] = {1, 1, 1, 1};
  int64_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t output[24] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int64_t output_exp[24] = {0, 0, 0, 0, 0, 0,
                            0, 1, 2, 3, 4, 0,
                            0, 5, 6, 7, 8, 0,
                            0, 0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS13) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2,2}};
  int32_t pad[4] = {1, 1, 1, 1};
  double input[8] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};
  double output[24] = {0.0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  double output_exp[24] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 1.1, 2.1, 3.1, 4.1, 0.0,
                           0.0, 5.1, 6.1, 7.1, 8.1, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool compare = CompareResult(output, output_exp, 24);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS14) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {}};
  int32_t pad[0];
  uint64_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t output[8] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  bool compare = CompareResult(output, output_exp, 8);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS15) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {5, 4}, {2, 2}};
  int32_t pad[4] = {1, 1, 1, 0};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[20] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[20] = {0, 0, 0, 0,
                             0, 1, 2, 3,
                             0, 4, 5, 6,
                             0, 7, 8, 9,
                             0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 20);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS16) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {4, 5}, {2, 2}};
  int32_t pad[4] = {0, 1, 1, 1};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[20] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[20] = {0, 1, 2, 3, 0,
                             0, 4, 5, 6, 0,
                             0, 7, 8, 9, 0,
                             0, 0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 20);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS17) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {5, 4}, {2, 2}};
  int32_t pad[4] = {1, 1, 0, 1};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[20] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[20] = {0, 0, 0, 0,
                             1, 2, 3, 0,
                             4, 5, 6, 0,
                             7, 8, 9, 0,
                             0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 20);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS18) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {4, 5}, {2, 2}};
  int32_t pad[4] = {1, 0, 1, 1};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[20] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[20] = {0, 0, 0, 0, 0,
                             0, 1, 2, 3, 0,
                             0, 4, 5, 6, 0,
                             0, 7, 8, 9, 0};
  bool compare = CompareResult(output, output_exp, 20);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS19) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {4, 4}, {2, 2}};
  int32_t pad[4] = {0, 1, 1, 0};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[16] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[20] = {0, 1, 2, 3,
                             0, 4, 5, 6,
                             0, 7, 8, 9,
                             0, 0, 0, 0};
  bool compare = CompareResult(output, output_exp, 16);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS20) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 4}, {2, 2}};
  int32_t pad[4] = {0, 0, 0, 1};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[12] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[12] = {1, 2, 3, 0,
                             4, 5, 6, 0,
                             7, 8, 9, 0};
  bool compare = CompareResult(output, output_exp, 12);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS21) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {4, 3}, {2, 2}};
  int32_t pad[4] = {1, 0, 0, 0};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[12] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[12] = {0, 0, 0,
                             1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
  bool compare = CompareResult(output, output_exp, 12);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_SUCCESS22) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {5, 3}, {2, 2}};
  int32_t pad[4] = {2, 0, 0, 0};
  uint64_t input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t output[15] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  uint64_t output_exp[15] = {0, 0, 0,
                             0, 0, 0,
                             1, 2, 3,
                             4, 5, 6,
                             7, 8, 9};
  bool compare = CompareResult(output, output_exp, 15);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_PAD_UT, DATA_FAILED1) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2, 2}};
  int32_t pad[4] = {1, 1, 1, 1};
  std::complex<float> input[8] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
  std::complex<float> output[8] = {{0, 0}};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PAD_UT, DATA_FAILED2) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2, 2}};
  int32_t pad[4] = {1, 1, 1, 1};
  std::complex<double> input[8] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
  std::complex<double> output[8] = {{0, 0}};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PAD_UT, DATA_FAILED3) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {2, 2}};
  int32_t pad[4] = {1, 1, 1, 1};
  std::complex<double> input[8] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
  std::complex<double> output[8] = {{0, 0}};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PAD_UT, DATA_FAILED4) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {}};
  int32_t pad[0];
  std::complex<double> input[8] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
  std::complex<double> output[8] = {{0, 0}};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PAD_UT, DATA_FAILED5) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2, 4}, {4, 6}, {}};
  uint32_t pad[4];
  uint64_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t output[8] = {0};
  vector<void *> datas = {(void *)input, (void *)output, (void *)pad};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}