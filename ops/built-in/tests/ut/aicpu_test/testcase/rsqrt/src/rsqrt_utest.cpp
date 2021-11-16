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

class TEST_RSQRT_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Rsqrt", "Rsqrt")                 \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})

//Read complex64 data
bool ReadFileComplex64(std::string file_name, std::complex<float> output[], uint64_t size) {
  size_t pos = file_name.find_last_of('.');
  if (pos == std::string::npos) {
    std::cout << "can not find extend name in file name: " << file_name << std::endl;
    return false;
  }
  std::string file_name_prefix = file_name.substr(0, pos);
  std::string real_file_name = file_name_prefix + "_real.txt";
  std::string imag_file_name = file_name_prefix + "_imag.txt";
  try {
    std::ifstream real_in_file{real_file_name};
    if (!real_in_file.is_open()) {
      std::cout << "open file: " << real_file_name << " failed." << std::endl;
      return false;
    }
    std::ifstream imag_in_file{imag_file_name};
    if (!imag_in_file.is_open()) {
      std::cout << "open file: " << imag_file_name << " failed." << std::endl;
      return false;
    }
    float real_data;
    float imag_data;
    uint64_t index = 0;
    while ((real_in_file >> real_data) && (imag_in_file >> imag_data)) {
      if (index >= size) {
        break;
      }
      output[index] = std::complex<float>(real_data, imag_data);
      index++;
    }
    real_in_file.close();
    imag_in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << real_file_name << " or " << imag_file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

//Read complex128 data
bool ReadFileComplex128(std::string file_name, std::complex<double> output[], uint64_t size) {
  size_t pos = file_name.find_last_of('.');
  if (pos == std::string::npos) {
    std::cout << "can not find extend name in file name: " << file_name << std::endl;
    return false;
  }
  std::string file_name_prefix = file_name.substr(0, pos);
  std::string real_file_name = file_name_prefix + "_real.txt";
  std::string imag_file_name = file_name_prefix + "_imag.txt";
  try {
    std::ifstream real_in_file{real_file_name};
    if (!real_in_file.is_open()) {
      std::cout << "open file: " << real_file_name << " failed." << std::endl;
      return false;
    }
    std::ifstream imag_in_file{imag_file_name};
    if (!imag_in_file.is_open()) {
      std::cout << "open file: " << imag_file_name << " failed." << std::endl;
      return false;
    }
    float real_data;
    float imag_data;
    uint64_t index = 0;
    while ((real_in_file >> real_data) && (imag_in_file >> imag_data)) {
      if (index >= size) {
        break;
      }
      output[index] = std::complex<double>(real_data, imag_data);
      index++;
    }
    real_in_file.close();
    imag_in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << real_file_name << " or " << imag_file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}
// read input and output data from files which generate by your python file
template<typename T1, typename T2>
void RunRsqrtKernel(vector<string> data_files,  
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
// read data from file for input
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size]; 
  vector<void *> datas = {(void *)input,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

   // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input;
  delete [] output;
  delete [] output_exp;
}
//read complex64 input and output data from files which generate by your python file
template<typename T1, typename T2>
void RunRsqrtKernelComplex64(vector<string> data_files,  
                          vector<DataType> data_types,
                          vector<vector<int64_t>> &shapes) {                   
// read data from file for input
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  bool status = ReadFileComplex64(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size]; 
  vector<void *> datas = {(void *)input,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

   // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFileComplex64(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input;
  delete [] output;
  delete [] output_exp;
}

//read complex128 input and output data from files which generate by your python file
template<typename T1, typename T2>
void RunRsqrtKernelComplex128(vector<string> data_files,  
                          vector<DataType> data_types,
                          vector<vector<int64_t>> &shapes) {                   
// read data from file for input
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  bool status = ReadFileComplex128(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size]; 
  vector<void *> datas = {(void *)input,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

   // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFileComplex128(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_1.txt",
                       "rsqrt/data/rsqrt_data_output1_1.txt"};
  RunRsqrtKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_2.txt",
                       "rsqrt/data/rsqrt_data_output1_2.txt"};
  RunRsqrtKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes =  {{5, 5}, {5, 5}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_3.txt",
                       "rsqrt/data/rsqrt_data_output1_3.txt"};
  RunRsqrtKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_FLOAT_MUTILCORE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{9, 1024}, {9, 1024}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_4.txt",
                       "rsqrt/data/rsqrt_data_output1_4.txt"};
  RunRsqrtKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_6.txt",
                       "rsqrt/data/rsqrt_data_output1_6.txt"};
  RunRsqrtKernelComplex64<std::complex<float>, std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_COMPLEX128__SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 5}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_7.txt",
                       "rsqrt/data/rsqrt_data_output1_7.txt"};
  RunRsqrtKernelComplex128<std::complex<double>, std::complex<double>>(files, data_types, shapes);
}

TEST_F(TEST_RSQRT_UT, DATA_TYPE_COMPLEX64_MUTILCORE_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{5, 1024}, {5, 1024}};
  vector<string> files{"rsqrt/data/rsqrt_data_input1_8.txt",
                       "rsqrt/data/rsqrt_data_output1_8.txt"};
  RunRsqrtKernelComplex64<std::complex<float>, std::complex<float>>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_RSQRT_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 10}, {2, 11}};
  double_t input[20] = {(double_t)1};
  double_t output[22] = {(double_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  double_t input[22] = {(double_t)1};
  float_t output[22] = {(float_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  double_t output[22] = {(double_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  double_t input[22] = {(double_t)1};
  vector<void *> datas = {(void *)input, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RSQRT_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool input[22] = {(bool)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
