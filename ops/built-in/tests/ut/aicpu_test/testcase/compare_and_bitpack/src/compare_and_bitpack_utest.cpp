/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

class TEST_COMPARE_AND_BITPACK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "CompareAndBitpack", "CompareAndBitpack") \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})              \
      .Input({"threshold", (data_types)[1], (shapes)[1], (datas)[1]})      \
      .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunCompareAndBitpackKernel(vector<string> data_files,
                                vector<DataType> data_types,
                                vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 input2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 output[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 output_exp[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = true;
  for (uint64_t i = 0; i < output_size; i++) {
    if (output[i] != output_exp[i]) {
      compare = false;
      printf("output[%lu] = %u, output_exp[%lu] = %u\n", i, output[i], i,
             output_exp[i]);
    }
  }
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{60, 80}, {}, {60, 10}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_1.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_1.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_1.txt"};
  RunCompareAndBitpackKernel<int32_t, int32_t, uint8_t>(files, data_types,
                                                        shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_UINT8};
  vector<vector<int64_t>> shapes = {{13, 80}, {}, {13, 10}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_2.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_2.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_2.txt"};
  RunCompareAndBitpackKernel<int64_t, int64_t, uint8_t>(files, data_types,
                                                        shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_UINT8};
  vector<vector<int64_t>> shapes = {{15, 320}, {}, {15, 40}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_3.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_3.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_3.txt"};
  RunCompareAndBitpackKernel<float, float, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_UINT8};
  vector<vector<int64_t>> shapes = {{64, 64, 128}, {}, {64, 64, 16}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_4.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_4.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_4.txt"};
  RunCompareAndBitpackKernel<double, double, uint8_t>(files, data_types,
                                                      shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{7, 160}, {}, {7, 20}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_5.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_5.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_5.txt"};
  RunCompareAndBitpackKernel<int8_t, int8_t, uint8_t>(files, data_types,
                                                      shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_BOOL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_UINT8};
  vector<vector<int64_t>> shapes = {{7, 160}, {}, {7, 20}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_6.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_6.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_6.txt"};
  RunCompareAndBitpackKernel<bool, bool, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_UINT8};
  vector<vector<int64_t>> shapes = {{12, 80}, {}, {12, 10}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_7.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_7.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_7.txt"};
  RunCompareAndBitpackKernel<int16_t, int16_t, uint8_t>(files, data_types,
                                                        shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_UINT8};
  vector<vector<int64_t>> shapes = {{120, 120}, {}, {120, 15}};
  vector<string> files{
      "compare_and_bitpack/data/compare_and_bitpack_data_input1_8.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_input2_8.txt",
      "compare_and_bitpack/data/compare_and_bitpack_data_output1_8.txt"};
  RunCompareAndBitpackKernel<Eigen::half, Eigen::half, uint8_t>(
      files, data_types, shapes);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_IS_SCALAR) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{}, {}, {12, 1}};
  int32_t input1[1] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  uint8_t output[12] = {(uint8_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{12, 10}, {}, {12, 1}};
  int32_t input1[120] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  uint8_t output[12] = {(uint8_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, THRESHOLD_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{12, 8}, {5}, {12, 1}};
  int32_t input1[96] = {(int32_t)1};
  int32_t input2[5] = {(int32_t)0};
  uint8_t output[12] = {(uint8_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_UINT8};
  vector<vector<int64_t>> shapes = {{12, 8}, {}, {12, 1}};
  int32_t input1[96] = {(int32_t)1};
  int64_t input2[1] = {(int64_t)0};
  uint8_t output[12] = {(uint8_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{12, 8}, {}, {12, 1}};
  uint8_t output[12] = {(uint8_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMPARE_AND_BITPACK_UT, INPUT_DTYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{12, 8}, {}, {12, 1}};
  uint32_t input1[96] = {(int32_t)1};
  uint32_t input2[1] = {(int64_t)0};
  uint8_t output[12] = {(uint8_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}