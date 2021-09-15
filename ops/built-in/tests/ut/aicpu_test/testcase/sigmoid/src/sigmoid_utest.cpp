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

class TEST_SIGMOID_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "Sigmoid", "Sigmoid")                         \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Output({"y", data_types[1], shapes[1], datas[1]})

bool ReadFileComplexFloatSigmoid(std::string file_name, std::complex<float> output[],
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

bool ReadFileComplexDoubleSigmoid(std::string file_name, std::complex<double> output[],
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

template<typename T1, typename T2>
void RunSigmoidKernel(vector<string> data_files,
                  vector<DataType> data_types,
                  vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  T2 output1[output1_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T2 output1_exp[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunSigmoidKernelComplexFloat(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = ReadFileComplexFloatSigmoid(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  std::complex<float> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<float> output1_exp[output1_size];
  status = ReadFileComplexFloatSigmoid(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunSigmoidKernelComplexDouble(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = ReadFileComplexDoubleSigmoid(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  std::complex<double> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<double> output1_exp[output1_size];
  status = ReadFileComplexDoubleSigmoid(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}


TEST_F(TEST_SIGMOID_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"sigmoid/data/sigmoid_data_input1_0.txt",
                       "sigmoid/data/sigmoid_data_output1_0.txt"};
  RunSigmoidKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{8, 4, 4}, {8,4,4}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_1.txt",
      "sigmoid/data/sigmoid_data_output_1_1.txt"};
  RunSigmoidKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 8, 4, 4}, {2, 8, 4, 4}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_2.txt",
      "sigmoid/data/sigmoid_data_output_1_2.txt"};
  RunSigmoidKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_3.txt",
      "sigmoid/data/sigmoid_data_output_1_3.txt"};
  RunSigmoidKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_4.txt",
      "sigmoid/data/sigmoid_data_output_1_4.txt"};
  RunSigmoidKernelComplexDouble(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_FLOAT_4D_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 8, 4, 4}, {2, 8, 4, 4}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_5.txt",
      "sigmoid/data/sigmoid_data_output_1_5.txt"};
  RunSigmoidKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_FLOATLess_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2}, {2}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_6.txt",
      "sigmoid/data/sigmoid_data_output_1_6.txt"};
  RunSigmoidKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_FLOATExp_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{8, 4, 4}, {8, 4, 4}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_7.txt",
      "sigmoid/data/sigmoid_data_output_1_7.txt"};
  RunSigmoidKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_FLOAT16LessExp_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_8.txt",
      "sigmoid/data/sigmoid_data_output_1_8.txt"};
  RunSigmoidKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_FLOATLessExp_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_9.txt",
      "sigmoid/data/sigmoid_data_output_1_9.txt"};
  RunSigmoidKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_DOUBLELessExp_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_10.txt",
      "sigmoid/data/sigmoid_data_output_1_10.txt"};
  RunSigmoidKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_COMPLEX64LessExp_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_11.txt",
      "sigmoid/data/sigmoid_data_output_1_11.txt"};
  RunSigmoidKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_FILE_DTYPE_COMPLEX128LessExp_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{1}, {1}};
  vector<string> files{
      "sigmoid/data/sigmoid_data_input_1_12.txt",
      "sigmoid/data/sigmoid_data_output_1_12.txt"};
  RunSigmoidKernelComplexDouble(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, DATA_TYPE_COMPLEX128Big_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"sigmoid/data/sigmoid_data_input1_13.txt",
                       "sigmoid/data/sigmoid_data_output1_13.txt"};
  RunSigmoidKernelComplexDouble(files, data_types, shapes);
}

TEST_F(TEST_SIGMOID_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  bool input1[250] = {(bool)1};
  bool output1[250] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOID_UT, INPUT_INT_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  int32_t input1[250] = {(int32_t)1};
  int32_t output1[250] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOID_UT, INPUT_INT64_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  int64_t input1[250] = {(int64_t)1};
  int64_t output1[250] = {(int64_t)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOID_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOID_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float input[22] = {(float)0};
  vector<void *> datas = {(void *)input, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}