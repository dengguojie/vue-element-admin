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

class TEST_ZEROSLIKE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "ZerosLike", "ZerosLike")                     \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Output({"y", data_types[1], shapes[1], datas[1]})


template <typename T>
void CalcExpectWithSameShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  int64_t input0_num = input0->NumElements();
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = static_cast<T>(0);
    }
}

bool ReadFileComplexFloatZerosLike(std::string file_name, std::complex<float> output[],
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

bool ReadFileComplexDoubleZerosLike(std::string file_name, std::complex<double> output[],
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
void RunZerosLikeKernel(vector<string> data_files,
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

template<typename T>
void RunZerosLikeKernel2(vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes) {
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T *input1 = new T[input1_size];
  SetRandomValue<T>(input1, input1_size);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  T *output_exp = new T[output_size];
    CalcExpectWithSameShape<T>(*node_def.get(), output_exp);


  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] output;
  delete [] output_exp;
}

void RunZerosLikeKernelComplexFloat(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = ReadFileComplexFloatZerosLike(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  std::complex<float> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<float> output1_exp[output1_size];
  status = ReadFileComplexFloatZerosLike(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

void RunZerosLikeKernelComplexDouble(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = ReadFileComplexDoubleZerosLike(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output1_size = CalTotalElements(shapes, 1);
  std::complex<double> output1[output1_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output1};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<double> output1_exp[output1_size];
  status = ReadFileComplexDoubleZerosLike(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  EXPECT_EQ(compare1, true);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12, 130}, {12, 130}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_0.txt",
      "zeroslike/data/zeroslike_data_output1_0.txt"};
  RunZerosLikeKernel<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{8, 4, 4}, {8, 4, 4}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_1.txt",
      "zeroslike/data/zeroslike_data_output1_1.txt"};
  RunZerosLikeKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 12, 30}, {7, 12, 30}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_2.txt",
      "zeroslike/data/zeroslike_data_output1_2.txt"};
  RunZerosLikeKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_3.txt",
      "zeroslike/data/zeroslike_data_output1_3.txt"};
  RunZerosLikeKernel<int8_t>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{12, 6}, {12, 6}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_4.txt",
      "zeroslike/data/zeroslike_data_output1_4.txt"};
  RunZerosLikeKernel<int16_t>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 12}, {6, 12}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_5.txt",
      "zeroslike/data/zeroslike_data_output1_5.txt"};
  RunZerosLikeKernel<int32_t>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input1_6.txt",
      "zeroslike/data/zeroslike_data_output1_6.txt"};
  RunZerosLikeKernel<int64_t>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}};
  RunZerosLikeKernel2<uint8_t>(data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}};
  RunZerosLikeKernel2<uint16_t>(data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}};
  RunZerosLikeKernel2<uint32_t>(data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}};
  RunZerosLikeKernel2<uint64_t>(data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input_1_7.txt",
      "zeroslike/data/zeroslike_data_output_1_7.txt"};
  RunZerosLikeKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input_1_8.txt",
      "zeroslike/data/zeroslike_data_output_1_8.txt"};
  RunZerosLikeKernelComplexDouble(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_FLOAT16Less_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2}, {2}};
  vector<string> files{
      "zeroslike/data/zeroslike_data_input_1_9.txt",
      "zeroslike/data/zeroslike_data_output_1_9.txt"};
  RunZerosLikeKernel<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_FILE_DTYPE_BOOL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  bool input1[250] = {(bool)1};
  bool output1[250] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_ZEROSLIKE_UT, INPUT_STRING_EXCEPTION) {
  vector<DataType> data_types = {DT_STRING, DT_STRING};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  float input1[250] = {(float)1};
  float output1[250] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}


TEST_F(TEST_ZEROSLIKE_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ZEROSLIKE_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float input[22] = {(float)0};
  vector<void *> datas = {(void *)input, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
