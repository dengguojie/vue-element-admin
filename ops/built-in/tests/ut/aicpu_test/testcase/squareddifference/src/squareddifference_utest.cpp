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

class TEST_SQUAREDDIFFERENCE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "SquaredDifference", "SquaredDifference")     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})                       \
      .Input({"x2", data_types[1], shapes[1], datas[1]})                       \
      .Output({"y", data_types[2], shapes[2], datas[2]})

bool ReadFileComplexFloatSquaredDifference(std::string file_name,
                         std::complex<float> output[],uint64_t size) {
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

bool ReadFileComplexDoubleSquaredDifference(std::string file_name, 
                          std::complex<double> output[], uint64_t size) {
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
void RunSquaredDifferenceKernel(vector<string> data_files,
                  vector<DataType> data_types,
                  vector<vector<int64_t>> &shapes) {
                  	
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
  
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T input2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 2);
  T output1[output1_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  T output1_exp[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunSquaredDifferenceKernelComplexFloat(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = 
      ReadFileComplexFloatSquaredDifference(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  std::complex<float> input2[input1_size];
  status = 
      ReadFileComplexFloatSquaredDifference(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 2);
  std::complex<float> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  std::complex<float> output1_exp[output1_size];
  status = 
      ReadFileComplexFloatSquaredDifference(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunSquaredDifferenceKernelComplexDouble(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = 
      ReadFileComplexDoubleSquaredDifference(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  std::complex<double> input2[input1_size];
  status = 
      ReadFileComplexDoubleSquaredDifference(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 2);
  std::complex<double> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  std::complex<double> output1_exp[output1_size];
  status = 
      ReadFileComplexDoubleSquaredDifference(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{128, 1024}, {128, 1024}, {128, 1024}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_0.txt",
      "squareddifference/data/squareddifference_data_input2_0.txt",
      "squareddifference/data/squareddifference_data_output1_0.txt"};
  RunSquaredDifferenceKernel<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{64, 1024}, {64, 1024}, {64, 1024}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_1.txt",
      "squareddifference/data/squareddifference_data_input2_1.txt",
      "squareddifference/data/squareddifference_data_output1_1.txt"};
  RunSquaredDifferenceKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 12, 30}, {7, 12, 30}, {7, 12, 30}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_2.txt",
      "squareddifference/data/squareddifference_data_input2_2.txt",
      "squareddifference/data/squareddifference_data_output1_2.txt"};
  RunSquaredDifferenceKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 12}, {6, 12}, {6, 12}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_3.txt",
      "squareddifference/data/squareddifference_data_input2_3.txt",
      "squareddifference/data/squareddifference_data_output1_3.txt"};
  RunSquaredDifferenceKernel<int32_t>(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}, {13, 10, 4}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_4.txt",
      "squareddifference/data/squareddifference_data_input2_4.txt",
      "squareddifference/data/squareddifference_data_output1_4.txt"};
  RunSquaredDifferenceKernel<int64_t>(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{64, 1024}, {64, 1024}, {64, 1024}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_5.txt",
      "squareddifference/data/squareddifference_data_input2_5.txt",
      "squareddifference/data/squareddifference_data_output1_5.txt"};
  RunSquaredDifferenceKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_FILE_DTYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}, {6, 6, 6}};
  vector<string> files{
      "squareddifference/data/squareddifference_data_input1_6.txt",
      "squareddifference/data/squareddifference_data_input2_6.txt",
      "squareddifference/data/squareddifference_data_output1_6.txt"};
  RunSquaredDifferenceKernelComplexDouble(files, data_types, shapes);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 1024}, {1,1024}, {4, 1024}};

  // read data from file for input1
  string data_path =
      ktestcaseFilePath + "squareddifference/data/squareddifference_data_input1_7.txt";
  constexpr uint64_t input1_size = 4 * 1024;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "squareddifference/data/squareddifference_data_input2_7.txt";
  constexpr uint64_t input2_size = 1024;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 4 * 1024;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "squareddifference/data/squareddifference_data_output1_7.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_X_NUM_ONE_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1}, {1,3}, {1,3}};

  // read data from file for input1
  string data_path =
      ktestcaseFilePath + "squareddifference/data/squareddifference_data_input1_8.txt";
  constexpr uint64_t input1_size = 1;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "squareddifference/data/squareddifference_data_input2_8.txt";
  constexpr uint64_t input2_size = 3;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 3;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "squareddifference/data/squareddifference_data_output1_8.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, BROADCAST_INPUT_Y_NUM_ONESUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1,3}, {1}, {1,3}};

  // read data from file for input1
  string data_path =
      ktestcaseFilePath + "squareddifference/data/squareddifference_data_input1_9.txt";
  constexpr uint64_t input1_size = 3;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "squareddifference/data/squareddifference_data_input2_9.txt";
  constexpr uint64_t input2_size = 1;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 3;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "squareddifference/data/squareddifference_data_output1_9.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}, {10, 5, 5}};
  bool input1[250] = {(bool)1};
  bool input2[250] = {(bool)1};
  bool output1[250] = {(bool)0};
  vector<void *> datas = {(void *)input1,(void *)input2, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  float output[22] = {(float)0};
  vector<void *> datas = {(void *)nullptr,(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQUAREDDIFFERENCE_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  float input1[22] = {(float)0};
  float input2[22] = {(float)0};
  vector<void *> datas = {(void *)input1,(void *)input2, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}