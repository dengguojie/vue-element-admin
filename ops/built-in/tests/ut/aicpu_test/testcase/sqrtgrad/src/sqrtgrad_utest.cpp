#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/kernel_util.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

bool CompareResultFloat16_SqrtGrad(Eigen::half output[],
                                   Eigen::half expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (Eigen::numext::abs(output[i] - expect_output[i]) > Eigen::half(1.0)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }

  return result;
}

class TEST_SQRTGRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "SqrtGrad", "SqrtGrad")           \
      .Input({"x1", (data_types)[0], (shapes)[0], (datas)[0]})     \
      .Input({"x2", (data_types)[1], (shapes)[1], (datas)[1]})     \
      .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})

bool ReadFileFloat_SqrtGrad(std::string file_name, std::complex<float> output[],
                            uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    for (uint64_t index = 0; index < size; ++index) {
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
      if (!flag) temp *= -1;
      output[index].imag(temp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool ReadFileDouble_SqrtGrad(std::string file_name,
                             std::complex<double> output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    for (uint64_t index = 0; index < size; ++index) {
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
      if (!flag) temp *= -1;
      output[index].imag(temp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T1, typename T2, typename T3>
void RunSqrtGradKernel(vector<string> data_files, vector<DataType> &data_types,
                       vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

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

template <typename T1, typename T2, typename T3>
void RunSqrtGradKernelFloat16(vector<string> data_files,
                              vector<DataType> &data_types,
                              vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResultFloat16_SqrtGrad(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT_SUCC_1D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15}, {15}, {15}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_1.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_1.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_1.txt"};
  RunSqrtGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT_SUCC_2D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 4}, {5, 4}, {5, 4}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_2.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_2.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_2.txt"};
  RunSqrtGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT_SUCC_3D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 4, 10}, {5, 4, 10}, {5, 4, 10}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_3.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_3.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_3.txt"};
  RunSqrtGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT_SUCC_4D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {
      {5, 4, 10, 2}, {5, 4, 10, 2}, {5, 4, 10, 2}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_4.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_4.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_4.txt"};
  RunSqrtGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT_SUCC_5D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {
      {5, 4, 10, 2, 2}, {5, 4, 10, 2, 2}, {5, 4, 10, 2, 2}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_5.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_5.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_5.txt"};
  RunSqrtGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT16_SUCC_1D) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12}, {12}, {12}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_6.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_6.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_6.txt"};
  RunSqrtGradKernelFloat16<Eigen::half, Eigen::half, Eigen::half>(
      files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT16_SUCC_2D) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{8, 1024}, {8, 1024}, {8, 1024}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_7.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_7.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_7.txt"};
  RunSqrtGradKernelFloat16<Eigen::half, Eigen::half, Eigen::half>(
      files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT16_SUCC_3D) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {4, 16, 4}, {4, 16, 4}, {4, 16, 4}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_8.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_8.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_8.txt"};
  RunSqrtGradKernelFloat16<Eigen::half, Eigen::half, Eigen::half>(
      files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {1, 4}, {1, 4}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_9.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_9.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_9.txt"};
  RunSqrtGradKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 1024, 4}, {2, 1024, 4}, {2, 1024, 4}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_10.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_10.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_10.txt"};
  RunSqrtGradKernel<std::complex<float>, std::complex<float>,
                    std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}, {6, 6, 6}};
  vector<string> files{"sqrtgrad/data/sqrtgrad_data_input1_11.txt",
                       "sqrtgrad/data/sqrtgrad_data_input2_11.txt",
                       "sqrtgrad/data/sqrtgrad_data_output1_11.txt"};
  RunSqrtGradKernel<std::complex<double>, std::complex<double>,
                    std::complex<double>>(files, data_types, shapes);
}
// exception instance

TEST_F(TEST_SQRTGRAD_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQRTGRAD_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  float output[22] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQRTGRAD_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  float input1[22] = {(float)1};
  Eigen::half input2[22] = {(Eigen::half)0};
  float output[22] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQRTGRAD_UT, DATA_TYPE_FLOAT16_DIV0) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  Eigen::half input1[22] = {(Eigen::half)1};
  Eigen::half input2[22] = {(Eigen::half)0};
  Eigen::half output[22] = {(Eigen::half)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}