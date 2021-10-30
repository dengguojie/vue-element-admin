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

class TEST_GREATER_EQUAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "GreaterEqual", "GreaterEqual")   \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunGreaterEqualKernel(vector<string> data_files,
                           vector<DataType> data_types,
                           vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  const uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);

  EXPECT_EQ(status, true);
  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  const uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  const uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {reinterpret_cast<void *>(input1),
                          reinterpret_cast<void *>(input2),
                          reinterpret_cast<void *>(output)};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_BOOL};
  vector<vector<int64_t>> shapes = {{6, 12}, {12}, {6, 12}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_1.txt",
                       "greater_equal/data/greater_equal_data_input2_1.txt",
                       "greater_equal/data/greater_equal_data_output1_1.txt"};
  RunGreaterEqualKernel<int8_t, int8_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{12, 6}, {12, 6}, {12, 6}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_2.txt",
                       "greater_equal/data/greater_equal_data_input2_2.txt",
                       "greater_equal/data/greater_equal_data_output1_2.txt"};
  RunGreaterEqualKernel<int16_t, int16_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_3.txt",
                       "greater_equal/data/greater_equal_data_input2_3.txt",
                       "greater_equal/data/greater_equal_data_output1_3.txt"};
  RunGreaterEqualKernel<int32_t, int32_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{11, 10, 4}, {11, 10, 4}, {11, 10, 4}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_4.txt",
                       "greater_equal/data/greater_equal_data_input2_4.txt",
                       "greater_equal/data/greater_equal_data_output1_4.txt"};
  RunGreaterEqualKernel<int64_t, int64_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_BOOL};
  vector<vector<int64_t>> shapes = {{6, 12}, {12}, {6, 12}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_5.txt",
                       "greater_equal/data/greater_equal_data_input2_5.txt",
                       "greater_equal/data/greater_equal_data_output1_5.txt"};
  RunGreaterEqualKernel<uint8_t, uint8_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{6, 12}, {12}, {6, 12}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_6.txt",
                       "greater_equal/data/greater_equal_data_input2_6.txt",
                       "greater_equal/data/greater_equal_data_output1_6.txt"};
  RunGreaterEqualKernel<uint16_t, uint16_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{7, 12, 10}, {12, 10}, {7, 12, 10}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_7.txt",
                       "greater_equal/data/greater_equal_data_input2_7.txt",
                       "greater_equal/data/greater_equal_data_output1_7.txt"};
  RunGreaterEqualKernel<uint32_t, uint32_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{12, 10}, {7, 12, 10}, {7, 12, 10}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_8.txt",
                       "greater_equal/data/greater_equal_data_input2_8.txt",
                       "greater_equal/data/greater_equal_data_output1_8.txt"};
  RunGreaterEqualKernel<uint64_t, uint64_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}, {15, 12, 30}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_9.txt",
                       "greater_equal/data/greater_equal_data_input2_9.txt",
                       "greater_equal/data/greater_equal_data_output1_9.txt"};
  RunGreaterEqualKernel<Eigen::half, Eigen::half, bool>(files, data_types,
                                                        shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {12, 30}, {15, 12, 30}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_10.txt",
                       "greater_equal/data/greater_equal_data_input2_10.txt",
                       "greater_equal/data/greater_equal_data_output1_10.txt"};
  RunGreaterEqualKernel<float, float, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_BOOL};
  vector<vector<int64_t>> shapes = {{7, 12, 30}, {30}, {7, 12, 30}};
  vector<string> files{"greater_equal/data/greater_equal_data_input1_11.txt",
                       "greater_equal/data/greater_equal_data_input2_11.txt",
                       "greater_equal/data/greater_equal_data_output1_11.txt"};
  RunGreaterEqualKernel<double, double, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, X_ONE_ELEMENT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1}, {1024, 1024}, {1024, 1024}};
  vector<string> files{
      "greater_equal/data/greater_equal_data_input1_x_one_elem.txt",
      "greater_equal/data/greater_equal_data_input2_x_one_elem.txt",
      "greater_equal/data/greater_equal_data_output1_x_one_elem.txt"};
  RunGreaterEqualKernel<int32_t, int32_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_EQUAL_UT, Y_ONE_ELEMENT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1024, 1024}, {1}, {1024, 1024}};
  vector<string> files{
      "greater_equal/data/greater_equal_data_input1_y_one_elem.txt",
      "greater_equal/data/greater_equal_data_input2_y_one_elem.txt",
      "greater_equal/data/greater_equal_data_output1_y_one_elem.txt"};
  RunGreaterEqualKernel<int32_t, int32_t, bool>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_GREATER_EQUAL_UT, INPUT_SHAPE_EXCEPTION1) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {static_cast<int32_t>(1)};
  int32_t input2[16] = {static_cast<int32_t>(0)};
  bool output[16] = {static_cast<bool>(0)};
  vector<void *> datas = {static_cast<void *>(input1),
                          static_cast<void *>(input2),
                          static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_EQUAL_UT, INPUT_SHAPE_EXCEPTION2) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 4}, {2, 6}, {2, 6}};
  int32_t input1[8] = {static_cast<int32_t>(1)};
  int32_t input2[12] = {static_cast<int32_t>(0)};
  bool output[12] = {static_cast<bool>(0)};
  vector<void *> datas = {static_cast<void *>(input1),
                          static_cast<void *>(input2),
                          static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_EQUAL_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {static_cast<int32_t>(1)};
  int64_t input2[22] = {static_cast<int64_t>(0)};
  bool output[22] = {static_cast<bool>(0)};
  vector<void *> datas = {static_cast<void *>(input1),
                          static_cast<void *>(input2),
                          static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_EQUAL_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool output[22] = {static_cast<bool>(0)};
  vector<void *> datas = {static_cast<void *>(nullptr),
                          static_cast<void *>(nullptr),
                          static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GREATER_EQUAL_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {static_cast<bool>(1)};
  bool input2[22] = {static_cast<bool>(0)};
  bool output[22] = {static_cast<bool>(0)};
  vector<void *> datas = {static_cast<void *>(input1),
                          static_cast<void *>(input2),
                          static_cast<void *>(output)};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}