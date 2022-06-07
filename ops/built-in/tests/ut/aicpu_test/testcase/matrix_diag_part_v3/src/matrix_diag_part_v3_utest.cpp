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
#include <Eigen/Core>
#include <iostream>
using namespace std;
using namespace aicpu;

class MatrixDiagPartV3Test : public testing::Test {};
#define CREATE_NODEDEF(shapes, data_types, datas, align)                       \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "MatrixDiagPartV3", "MatrixDiagPartV3")       \
      .Input({"input", data_types[0], shapes[0], datas[0], FORMAT_ND})         \
      .Input({"k", data_types[1], shapes[1], datas[1], FORMAT_ND})             \
      .Input({"padding_value", data_types[2], shapes[2], datas[2], FORMAT_ND}) \
      .Output({"diagonal", data_types[3], shapes[3], datas[3], FORMAT_ND})     \
      .Attr("align", align)

template <typename T1, typename T2, typename T3, typename T4>
void RunMatrixDiagPartV3Kernel(vector<string> data_files, vector<DataType> data_types, vector<vector<int64_t>>& shapes,
                               const string& alignvalue) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1* input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2* input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T3* input3 = new T3[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);
  uint64_t output_size = CalTotalElements(shapes, 3);
  T4* output = new T4[output_size];
  vector<void*> datas = {(void*)input1, (void*)input2, (void*)input3, (void*)output};
  string align = alignvalue;
  CREATE_NODEDEF(shapes, data_types, datas, align);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  data_path = ktestcaseFilePath + data_files[3];
  T4* output_exp = new T4[output_size];
  // status = ReadFile(data_path, output_exp, output_size);
  // EXPECT_EQ(status, true);

  // bool compare = CompareResult(output, output_exp, output_size);
  // EXPECT_EQ(compare, true);

  delete[] input1;
  delete[] input2;
  delete[] input3;
  delete[] output;
  delete[] output_exp;
}

// INT32
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_INT32) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 5, 3, 2}, {2}, {1}, {4, 5, 2}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_1.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_1.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_1.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_1.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<int32_t, int32_t, int32_t, int32_t>(files, data_types, shapes, align);
}

TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_FLOAT) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {2}, {1}, {5, 2}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_2.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_2.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_2.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_2.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<float_t, int32_t, float_t, float_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_DOUBLE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {2}, {1}, {5, 2}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_3.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_3.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_3.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_3.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<double_t, int32_t, double_t, double_t>(files, data_types, shapes, align);
}

TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_COMPLEX64) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {2}, {1}, {5, 2}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_4.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_4.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_4.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_4.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<std::complex<std::float_t>, int32_t, std::complex<std::float_t>,
                            std::complex<std::float_t>>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_INT64) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {2}, {1}, {5, 2}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_5.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_5.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_5.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_5.txt"};
  string align = "LEFT_LEFT";
  RunMatrixDiagPartV3Kernel<int64_t, int32_t, int64_t, int64_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_FLOAT16) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{6, 3, 4}, {2}, {1}, {6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_6.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_6.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_6.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_6.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<Eigen::half, int32_t, Eigen::half, Eigen::half>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_INT16) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{5, 6, 3, 4}, {2}, {1}, {5, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_7.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_7.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_7.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_7.txt"};

  string align = "RIGHT_RIGHT";
  RunMatrixDiagPartV3Kernel<int16_t, int32_t, int16_t, int16_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_INT8) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{5, 6, 3, 4}, {2}, {1}, {5, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_8.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_8.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_8.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_8.txt"};
  string align = "LEFT_RIGHT";
  RunMatrixDiagPartV3Kernel<int8_t, int32_t, int8_t, int8_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_UINT8) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{2, 6, 3, 4}, {2}, {1}, {2, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_9.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_9.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_9.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_9.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<uint8_t, int32_t, uint8_t, uint8_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_UINT16) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{2, 6, 3, 4}, {2}, {1}, {2, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_10.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_10.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_10.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_10.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<uint16_t, int32_t, uint16_t, uint16_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_UINT32) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2, 6, 3, 4}, {2}, {1}, {2, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_11.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_11.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_11.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_11.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<uint32_t, int32_t, uint32_t, uint32_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_UINT64) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 6, 3, 4}, {2}, {1}, {2, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_12.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_12.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_12.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_12.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<uint64_t, int32_t, uint64_t, uint64_t>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, matrix_diag_part_v3_test_COMPLEX128) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 6, 3, 4}, {2}, {1}, {2, 6, 3}};
  vector<string> files{"matrix_diag_part_v3/data/matrix_diag_part_v3_data_input1_13.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input2_13.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_input3_13.txt",
                       "matrix_diag_part_v3/data/matrix_diag_part_v3_data_output1_13.txt"};
  string align = "RIGHT_LEFT";
  RunMatrixDiagPartV3Kernel<std::complex<std::double_t>, int32_t, std::complex<std::double_t>,
                            std::complex<std::double_t>>(files, data_types, shapes, align);
}
TEST_F(MatrixDiagPartV3Test, ALIGN_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {1}, {1}, {5, 2}};
  string align = "LFT_LEFT";
  int64_t input[30] = {(int64_t)1};
  int32_t k[1] = {(int32_t)0};
  int64_t padding_value[1] = {(int64_t)0};
  vector<void*> datas = {(void*)input, (void*)k, (void*)padding_value};
  CREATE_NODEDEF(shapes, data_types, datas, align);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(MatrixDiagPartV3Test, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {1}, {1}, {5, 2}};
  string align = "LEFT_LEFT";
  int64_t input[30] = {(bool)1};
  int32_t k[1] = {(int32_t)0};
  int64_t padding_value[1] = {(bool)0};
  vector<void*> datas = {(void*)input, (void*)k, (void*)padding_value};
  CREATE_NODEDEF(shapes, data_types, datas, align);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(MatrixDiagPartV3Test, Exception_k_value) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{5, 3, 2}, {2}, {1}, {5, 2}};
  string align = "LEFT_LEFT";
  int64_t input[30] = {(int64_t)1};
  int32_t k[2] = {1, 0};
  int64_t padding_value[1] = {(int64_t)0};
  vector<void*> datas = {(void*)input, (void*)k, (void*)padding_value};
  CREATE_NODEDEF(shapes, data_types, datas, align);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
