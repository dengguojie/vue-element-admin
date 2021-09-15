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

class TEST_SIGN_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "Sign", "Sign")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Output({"y", data_types[1], shapes[1], datas[1]})

bool ReadFileComplexFloatSign(std::string file_name, std::complex<float> output[],
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

bool ReadFileComplexDoubleSign(std::string file_name, std::complex<double> output[],
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

template<typename T>
void RunSignKernel(vector<string> data_files,
                  vector<DataType> data_types,
                  vector<vector<int64_t>> &shapes) {
                  	
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  T output1[output1_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T output1_exp[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunSignKernelComplexFloat(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = ReadFileComplexFloatSign(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  std::complex<float> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<float> output1_exp[output1_size];
  status = ReadFileComplexFloatSign(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunSignKernelComplexDouble(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = ReadFileComplexDoubleSign(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  std::complex<double> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<double> output1_exp[output1_size];
  status = ReadFileComplexDoubleSign(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12, 130}, {12, 130}};
  vector<string> files{
      "sign/data/sign_data_input1_0.txt",
      "sign/data/sign_data_output1_0.txt"};
  RunSignKernel<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{64, 1024}, {64, 1024}};
  vector<string> files{
      "sign/data/sign_data_input1_1.txt",
      "sign/data/sign_data_output1_1.txt"};
  RunSignKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 12, 30}, {7, 12, 30}};
  vector<string> files{
      "sign/data/sign_data_input1_2.txt",
      "sign/data/sign_data_output1_2.txt"};
  RunSignKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 12}, {6, 12}};
  vector<string> files{
      "sign/data/sign_data_input1_3.txt",
      "sign/data/sign_data_output1_3.txt"};
  RunSignKernel<int32_t>(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}};
  vector<string> files{
      "sign/data/sign_data_input1_4.txt",
      "sign/data/sign_data_output1_4.txt"};
  RunSignKernel<int64_t>(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{64, 1024}, {64, 1024}};
  vector<string> files{
      "sign/data/sign_data_input_1_5.txt",
      "sign/data/sign_data_output_1_5.txt"};
  RunSignKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}};
  vector<string> files{
      "sign/data/sign_data_input_1_6.txt",
      "sign/data/sign_data_output_1_6.txt"};
  RunSignKernelComplexDouble(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_FLOAT16Less_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2}, {2}};
  vector<string> files{
      "sign/data/sign_data_input_1_7.txt",
      "sign/data/sign_data_output_1_7.txt"};
  RunSignKernel<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_COMPLEX64Less_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2}, {2}};
  vector<string> files{
      "sign/data/sign_data_input_1_8.txt",
      "sign/data/sign_data_output_1_8.txt"};
  RunSignKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_FILE_DTYPE_COMPLEX64Zero_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2}, {2}};
  vector<string> files{
      "sign/data/sign_data_input_1_9.txt",
      "sign/data/sign_data_output_1_9.txt"};
  RunSignKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_SIGN_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  bool input1[250] = {(bool)1};
  bool output1[250] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGN_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGN_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float input[22] = {(float)0};
  vector<void *> datas = {(void *)input, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}