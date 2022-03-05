/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 */

#include <gtest/gtest.h>
#ifndef private
#define private public
#define protected public
#endif
#include <Eigen/Core>
#include <iostream>

#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_COMPARE_AND_BITPACK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                             \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();            \
  NodeDefBuilder(node_def.get(), "CompareAndBitpack", "CompareAndBitpack")    \
      .Input({"x", data_types[0], shapes[0], datas[0]})                       \
      .Input({"threshold", data_types[1], shapes[1], datas[1]})               \
      .Output({"output", data_types[2], shapes[2], datas[2]})         

/*
template <typename T>
void RunTestCompareAndBitpack(string test_case_no, vector<DataType> data_types, T thresh) {
  // get input shape
  vector<int64_t> inputshape_data;
  string input_shape_path =
      ktestcaseFilePath + "compare_and_bitpack/data/compare_and_bitpack_input_data_shape_" + test_case_no + ".txt";
  // calc output shape
  uint64_t dims = inputshape_data.size();
  uint64_t last_dim = inputshape_data[dims - 1];
  vector<int64_t> outputshape_data;
  outputshape_data = inputshape_data;
  outputshape_data[dims - 1] = last_dim / 8;

  // get input data
  uint64_t inputdata_size = 1;
  for (uint64_t i = 0; i < dims; i++) {
    inputdata_size *= inputshape_data[i];
  }
  T input_data[inputdata_size] = {0};
  string input_data_path = ktestcaseFilePath + "compare_and_bitpack/data/compare_and_bitpack_input_data_" + test_case_no + ".txt";
  uint64_t outputdata_size = 1;
  for (uint64_t i = 0; i < outputshape_data.size(); i++) {
    outputdata_size *= outputshape_data[i];
  }
  uint8_t output_data[outputdata_size] = {0};

  // compute
  vector<vector<int64_t>> shapes = {inputshape_data, {}, outputshape_data};
  vector<void*> datas = {(void*)input_data, (void *)&thresh, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK)

  // compare the result
  string output_data_path = ktestcaseFilePath + "compare_and_bitpack/data/compare_and_bitpack_output_data_" + test_case_no + ".txt";
  uint8_t output_exp[outputdata_size] = {0};
  bool compare = CompareResult(output_data, output_exp, outputdata_size);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_FLOAT_SUCCESS) {
  RunTestCompareAndBitpack<float>("1", {DT_FLOAT, DT_FLOAT, DT_UINT8}, 20.0);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_DOUBLE_SUCCESS) {
  RunTestCompareAndBitpack<double>("2", {DT_DOUBLE, DT_DOUBLE, DT_UINT8}, 20.0);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT8_SUCCESS) {
  RunTestCompareAndBitpack<int8_t>("3", {DT_INT8, DT_INT8, DT_UINT8}, 20);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT16_SUCCESS) {
  RunTestCompareAndBitpack<int16_t>("4", {DT_INT16, DT_INT16, DT_UINT8}, 20);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT32_SUCCESS) {
  RunTestCompareAndBitpack<int32_t>("5", {DT_INT32, DT_INT32, DT_UINT8}, 20);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT64_SUCCESS) {
  RunTestCompareAndBitpack<int64_t>("6", {DT_INT64, DT_INT64, DT_UINT8}, 20);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT64_SUCCESS1) {
  RunTestCompareAndBitpack<int64_t>("7", {DT_INT64, DT_INT64, DT_UINT8}, 20);
}
*/
TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_COMPLEX64_SUCCESS) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  std::complex<float>  input0[16] = {2.6, 4.9, 3.5, 2.1, 1.9, 1.8, 4.5, 2.9, 2.6, 1.2, 1.6, 4.1, 1.5, 2.2, 4.3, 1.5};
  std::complex<float>  input1 = 3.2;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_FLOAT16_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  Eigen::half input0[16] = {(Eigen::half)2.6, (Eigen::half)4.9, (Eigen::half)3.5, (Eigen::half)2.1,
                            (Eigen::half)1.9, (Eigen::half)1.8, (Eigen::half)4.5, (Eigen::half)2.9,
                            (Eigen::half)2.6, (Eigen::half)1.2, (Eigen::half)1.6, (Eigen::half)4.1,
                            (Eigen::half)1.5, (Eigen::half)2.2, (Eigen::half)4.3, (Eigen::half)1.5};
  Eigen::half input1 = (Eigen::half)3.2;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_BOOl_SUCCESS) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  bool input0[16] = {false, true, true, false, false, false, true, false,
                     false, false, false, true, false, false, true, false};
  bool input1 = true;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, false);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_FLOAT_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  float input0[16] = {2.6, 4.9, 3.5, 2.1, 1.9, 1.8, 4.5, 2.9, 2.6, 1.2, 1.6, 4.1, 1.5, 2.2, 4.3, 1.5};
  float input1 = 3.2;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT8_SUCCESS) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  int8_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4,  1};
  int8_t input1 = 3;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT16_SUCCESS) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  int16_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4,  1};
  int16_t input1 = 3;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT32_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  int32_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4,  1};
  int32_t input1 = 3;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_INT64_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  int64_t input0[16] = {2, 4, 4, 2, 1, 1, 4, 2, 2, 1, 1, 4, 1, 2, 4,  1};
  int64_t input1 = 3;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DOUBLE__SUCCESS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                       4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                       1.53567731, 2.29360969, 4.3461758,  1.57004211};
  double input1 = 3.20928929;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint8_t output_exp[2] = {98, 18};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_0_EMPTY) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                       4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                       1.53567731, 2.29360969, 4.3461758,  1.57004211};
  double input1 = 3.20928929;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_1_EMPTY) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                       4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                       1.53567731, 2.29360969, 4.3461758,  1.57004211};
  double input1 = 3.20928929;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, OUTPUT_EMPTY) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                       4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                       1.53567731, 2.29360969, 4.3461758,  1.57004211};
  double input1 = 3.20928929;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)nullptr, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_1_NOT_SCALAR) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {1, 2}, {1, 2}};
  double input0[16] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                       4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                       1.53567731, 2.29360969, 4.3461758,  1.57004211};
  double input1[2] = {3.20928929, 20928929};
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_0_SCALAR) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{}, {}, {1, 2}};
  double input0 = 2.69160455;
  double input1 = 3.20928929;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)&input0, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_0_DIM_ERROR) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 16}, {}, {1, 2}};
  double input0[17] = {2.69160455, 4.97408142, 3.50928929, 2.11724898, 1.90514646, 1.87424379,
                       4.59832463, 2.95999397, 2.63020151, 1.23305568, 1.66515593, 4.14401462,
                       1.53567731, 2.29360969, 4.3461758,  1.57004211, 2.57004211};
  double input1 = 3.20928929;
  uint8_t output[2] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)&input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}