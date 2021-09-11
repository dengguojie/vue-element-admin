#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include <cmath>

#undef private
#undef protected
#include "Eigen/Core"
#include <string>
#include <iostream>
#include <sstream>

using namespace std;
using namespace aicpu;

class TEST_LOG_MATRIX_DETERMINANT_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "LogMatrixDeterminant",           \
                 "LogMatrixDeterminant")                           \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"sign", data_types[1], shapes[1], datas[1]})        \
      .Output({"y", data_types[2], shapes[2], datas[2]});

bool ReadFileComplexFloatLMD(std::string file_name, std::complex<float> output[],
                          uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    uint64_t index = 0;
    while (!in_file.eof()) {
      string s, s1, s2;
      stringstream ss, sss;
      string ::size_type n1, n2, n3;
      bool flag = true;

      getline(in_file, s);
      n1 = s.find("(", 0);
      n2 = s.find("+", 0);
      if (n2 == string::npos) {
        n2 = s.find("-", n1 + 2);
        flag = false;
      }
      n3 = s.find("j", 0);
      s1 = s.substr(n1 + 1, n2 - n1 - 1);
      s2 = s.substr(n2 + 1, n3 - n2 - 1);

      float temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
      if (!flag)
        temp *= -1;
      output[index].imag(temp);
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool ReadFileComplexDoubleLMD(std::string file_name, std::complex<double> output[],
                           uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    uint64_t index = 0;
    while (!in_file.eof()) {
      string s, s1, s2;
      stringstream ss, sss;
      string ::size_type n1, n2, n3;
      bool flag = true;

      getline(in_file, s);
      n1 = s.find("(", 0);
      n2 = s.find("+", 0);
      if (n2 == string::npos) {
        n2 = s.find("-", n1 + 2);
        flag = false;
      }
      n3 = s.find("j", 0);
      s1 = s.substr(n1 + 1, n2 - n1 - 1);
      s2 = s.substr(n2 + 1, n3 - n2 - 1);

      double temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
      if (!flag)
        temp *= -1;
      output[index].imag(temp);
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

void RunLogMatrixDeterminantKernelFloat(vector<string> data_files,
                                        vector<DataType> data_types,
                                        vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  float input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  float output1[output1_size];

  uint64_t output2_size = CalTotalElements(shapes, 2);
  float output2[output2_size];
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  float output1_exp[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[2];
  float output2_exp[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);

  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  EXPECT_EQ(compare2, true);
}

void RunLogMatrixDeterminantKernelDouble(vector<string> data_files,
                                         vector<DataType> data_types,
                                         vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  double input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  double output1[output1_size];

  uint64_t output2_size = CalTotalElements(shapes, 2);
  double output2[output2_size];
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  double output1_exp[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[2];
  double output2_exp[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);

  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  EXPECT_EQ(compare2, true);
}

template <typename T1, typename T2, typename T3>
void RunLogMatrixDeterminantKernel3(vector<string> data_files,
                                    vector<DataType> data_types,
                                    vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size];
  bool status = ReadFileComplexFloatLMD(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  T2 output1[output1_size];

  uint64_t output2_size = CalTotalElements(shapes, 2);
  T3 output2[output2_size];
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T2 output1_exp[output1_size];
  status = ReadFileComplexFloatLMD(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[2];
  T3 output2_exp[output2_size];
  status = ReadFileComplexFloatLMD(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);

  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  EXPECT_EQ(compare2, true);
}

template <typename T1, typename T2, typename T3>
void RunLogMatrixDeterminantKernel4(vector<string> data_files,
                                    vector<DataType> data_types,
                                    vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size];
  bool status = ReadFileComplexDoubleLMD(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  T2 output1[output1_size];

  uint64_t output2_size = CalTotalElements(shapes, 2);
  T3 output2[output2_size];
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T2 output1_exp[output1_size];
  status = ReadFileComplexDoubleLMD(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[2];
  T3 output2_exp[output2_size];
  status = ReadFileComplexDoubleLMD(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);

  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  EXPECT_EQ(compare2, true);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}};
  vector<string> data_files{
      "log_matrix_determinant/data/log_matrix_determinant_data_input_1_1.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_1_1.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_2_1.txt"};
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  float input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = 1;
  float output1[output1_size];

  uint64_t output2_size = 1;
  float output2[output2_size];
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  float output1_exp[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[2];
  float output2_exp[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);

  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  EXPECT_EQ(compare2, true);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{8, 16, 16}, {8}, {8}};
  vector<string> files{
      "log_matrix_determinant/data/log_matrix_determinant_data_input_1_2.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_1_2.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_2_2.txt"};
  RunLogMatrixDeterminantKernelDouble(files, data_types, shapes);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_FILE_DTYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10}, {10}};
  vector<string> files{
      "log_matrix_determinant/data/log_matrix_determinant_data_input_1_3.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_1_3.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_2_3.txt"};
  RunLogMatrixDeterminantKernel3<std::complex<float>, std::complex<float>,
                                 std::complex<float>>(files, data_types,
                                                      shapes);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_FILE_DTYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6}, {6}};
  vector<string> files{
      "log_matrix_determinant/data/log_matrix_determinant_data_input_1_4.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_1_4.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_2_4.txt"};
  RunLogMatrixDeterminantKernel4<std::complex<double>, std::complex<double>,
                                 std::complex<double>>(files, data_types,
                                                       shapes);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_FILE_DTYPE_FLOAT_4D_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 8, 4, 4}, {2, 8}, {2, 8}};
  vector<string> files{
      "log_matrix_determinant/data/log_matrix_determinant_data_input_1_5.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_1_5.txt",
      "log_matrix_determinant/data/log_matrix_determinant_data_output_2_5.txt"};
  RunLogMatrixDeterminantKernelFloat(files, data_types, shapes);
}

// exception
TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10}, {10}};
  bool input1[250] = {(bool)1};
  bool output1[10] = {(bool)0};
  bool output2[10] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_INT_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10}, {10}};
  int32_t input1[250] = {(int32_t)1};
  int32_t output1[10] = {(int32_t)0};
  int32_t output2[10] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_INT64_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10}, {10}};
  int64_t input1[250] = {(int64_t)1};
  int64_t output1[10] = {(int64_t)0};
  int64_t output2[10] = {(int64_t)0};
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LOG_MATRIX_DETERMINANT_UT, INPUT_SHAPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{10, 5, 4}, {10}, {10}};
  float input1[250] = {(float)1};
  float output1[10] = {(float)0};
  float output2[10] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}