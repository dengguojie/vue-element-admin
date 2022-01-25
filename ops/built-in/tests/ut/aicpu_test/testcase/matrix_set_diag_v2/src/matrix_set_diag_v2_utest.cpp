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
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_MATRIX_SET_DIAG_V2_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                      \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  NodeDefBuilder(node_def.get(), "MatrixSetDiagV2", "MatrixSetDiagV2") \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})          \
      .Input({"diagonal", (data_types)[1], (shapes)[1], (datas)[1]})   \
      .Input({"k", (data_types)[2], (shapes)[2], (datas)[2]})          \
      .Output({"y", (data_types)[3], (shapes)[3], (datas)[3]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3, typename T4>
void RunMatrixSetDiagV2Kernel(vector<string> data_files,
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

  // read data from file for input3
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T3 input3[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 3);
  T4 output[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[3];
  T4 output_exp[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult<T4>(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_1.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_1.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_1.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_1.txt"};
  RunMatrixSetDiagV2Kernel<int8_t, int8_t, int32_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {2}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_2.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_2.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_2.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_2.txt"};
  RunMatrixSetDiagV2Kernel<int16_t, int16_t, int32_t, int16_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_3.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_3.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_3.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_3.txt"};
  RunMatrixSetDiagV2Kernel<int32_t, int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {2}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_4.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_4.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_4.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_4.txt"};
  RunMatrixSetDiagV2Kernel<int64_t, int64_t, int32_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {6, 7}, {}, {6, 7, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_5.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_5.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_5.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_5.txt"};
  RunMatrixSetDiagV2Kernel<uint8_t, uint8_t, int32_t, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {6, 7}, {2}, {6, 7, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_6.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_6.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_6.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_6.txt"};
  RunMatrixSetDiagV2Kernel<uint16_t, uint16_t, int32_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{6, 8, 7}, {6, 7}, {}, {6, 8, 7}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_7.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_7.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_7.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_7.txt"};
  RunMatrixSetDiagV2Kernel<uint32_t, uint32_t, int32_t, uint32_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{6, 8, 7}, {6, 7}, {2}, {6, 8, 7}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_8.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_8.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_8.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_8.txt"};
  RunMatrixSetDiagV2Kernel<uint64_t, uint64_t, int32_t, uint64_t>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {2}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_9.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_9.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_9.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_9.txt"};
  RunMatrixSetDiagV2Kernel<Eigen::half, Eigen::half, int32_t, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {2}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_10.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_10.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_10.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_10.txt"};
  RunMatrixSetDiagV2Kernel<float, float, int32_t, float>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{64, 64, 64}, {64, 64}, {2}, {64, 64, 64}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_11.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_11.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_11.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_11.txt"};
  RunMatrixSetDiagV2Kernel<double, double, int32_t, double>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_12.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_12.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_12.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_12.txt"};
  RunMatrixSetDiagV2Kernel<std::complex<float>, std::complex<float>, int32_t, std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {6, 8}, {}, {6, 8, 8}};
  vector<string> files{"matrix_set_diag_v2/data/matrix_set_diag_v2_data_input1_13.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input2_13.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_input3_13.txt",
                       "matrix_set_diag_v2/data/matrix_set_diag_v2_data_output1_13.txt"};
  RunMatrixSetDiagV2Kernel<std::complex<double>, std::complex<double>, int32_t, std::complex<double>>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, NOT_LEADING_DIAGONAL_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3}, {}, {2, 3, 4}};
  int32_t input1[24] = {(int32_t)0};
  int32_t input2[6] = {1, 2, 3, 4, 5, 6};
  int32_t input3[1] = {(int32_t)1};
  int32_t output[24] = {(int32_t)0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int32_t output_exp[24] = {0, 1, 0, 0,
                            0, 0, 2, 0,
                            0, 0, 0, 3,
                            0, 4, 0, 0,
                            0, 0, 5, 0,
                            0, 0, 0, 6};
  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_BAND_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 3, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[24] = {1, 2, 3, 4,
                        5, 6, 7, 8,
                        9,10,11,12,
                        10,9, 8, 7,
                        6, 5, 4, 3,
                        2, 1, 0,-1};
  int32_t input3[2] = {-1, 1};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int32_t output_exp[32] = {5, 1, 0, 0,
                            9, 6, 2, 0,
                            0,10, 7, 3,
                            0, 0,11, 8, 
                            10,6, 0, 0,
                            2, 9, 5, 0,
                            0, 1, 8, 4,
                            0, 0, 0, 7};
  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[8] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10}, {10}, {}, {10}};
  int32_t input1[10] = {(int32_t)0};
  int32_t input2[10] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[10] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, K_DATASIZE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {3}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[8] = {(int32_t)0};
  int32_t input3[3] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, K_VALUE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[8] = {(int32_t)0};
  int32_t input3[3] = {2, 1};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_DIM_K1_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[32] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_DIM_K2_EQUAL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[32] = {(int32_t)0};
  int32_t input3[2] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_DIM_K2_NOT_EQUAL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[8] = {(int32_t)0};
  int32_t input3[2] = {1, 2};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_SHAPE_K1_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {3, 4}, {}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[12] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_SHAPE_K2_EQUAL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {3, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[12] = {(int32_t)0};
  int32_t input3[2] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DIAGONAL_SHAPE_K2_NOT_EQUAL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {3, 2, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[24] = {(int32_t)0};
  int32_t input3[2] = {0, 1};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}


TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, MAX_DIAG_LEN_K1_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 5}, {}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[10] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, MAX_DIAG_LEN_K2_EQUAL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 3}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[6] = {(int32_t)0};
  int32_t input3[2] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, MAX_DIAG_LEN_K2_NOT_EQUAL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 2, 5}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[20] = {(int32_t)0};
  int32_t input3[2] = {0, 1};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, NUM_DIAGS_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 3, 4}, {2}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[24] = {(int32_t)0};
  int32_t input3[2] = {0, 1};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATATSIZE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {}, {2, 4, 5}};
  int32_t input1[32] = {(int32_t)0};
  int32_t input2[8] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[40] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, DATATYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {}, {2, 4, 4}};
  int32_t input1[32] = {(int32_t)0};
  float input2[8] = {(float)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[32] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_SET_DIAG_V2_UT, INPUT_DATATYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4}, {}, {2, 4, 4}};
  bool input1[32] = {(bool)0};
  bool input2[8] = {(bool)0};
  int32_t input3[1] = {(int32_t)0};
  bool output[32] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}