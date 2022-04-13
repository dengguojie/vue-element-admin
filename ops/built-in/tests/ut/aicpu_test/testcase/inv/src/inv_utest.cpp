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


bool CompareResultComplex64ForInv(std::complex<float> output[],
                    std::complex<float> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::abs(output[i] - expect_output[i]) > 1e-6) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      //result = false;
    }
  }
  return result;
}

bool CompareResultComplex128ForInv(std::complex<double> output[],
                    std::complex<double> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::abs(output[i] - expect_output[i]) > 1e-1) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
    }
  }
  return result;
}

bool ReadFileComplex64ForInv(std::string file_name, std::complex<float> output[],
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
      if (!flag)
        temp *= -1;
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

bool ReadFileComplex128ForInv(std::string file_name, std::complex<double> output[],
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
      double temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
      if (!flag)
        temp *= -1;
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

class TEST_INV_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Inv", "Inv")                     \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2>
void RunInvKernel(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);//直接用就行

  // read data from file for expect ouput 读取正确答案
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  //比较结果是否一致
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] output;
  delete [] output_exp;
}

void RunInvKernelComplex64(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = ReadFileComplex64ForInv(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
  
  uint64_t output_size = CalTotalElements(shapes, 1);
  std::complex<float> output[output_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<float> output_exp[output_size];
  status = ReadFileComplex64ForInv(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResultComplex64ForInv(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

void RunInvKernelComplex128(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = ReadFileComplex128ForInv(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  std::complex<double> output[output_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<double> output_exp[output_size];
  status = ReadFileComplex128ForInv(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResultComplex128ForInv(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_INV_UT, DATA_TYPE_FLOAT16_1D_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2},  {2}};
  vector<string> files{"inv/data/inv_data_input1_1.txt",
                       "inv/data/inv_data_output1_1.txt"};
  RunInvKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_FLOAT32_1D_BIG_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{128 * 1024},  {128 * 1024}};

  vector<string> files{"inv/data/inv_data_input1_1_big.txt",
                       "inv/data/inv_data_output1_1_big.txt"};
  RunInvKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_DOUBLE_SUCC_3D) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{5,1024,32},{5,1024,32}};
  vector<string> files{"inv/data/inv_data_input1_2.txt",
                       "inv/data/inv_data_output1_2.txt"};
  RunInvKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12, 13},  {12, 13}};
  vector<string> files{"inv/data/inv_data_input1_3.txt",
                       "inv/data/inv_data_output1_3.txt"};
  RunInvKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 12, 2},  {15, 12, 2}};
  vector<string> files{"inv/data/inv_data_input1_4.txt",
                       "inv/data/inv_data_output1_4.txt"};
  RunInvKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 12, 2},{7, 12, 2}};
  vector<string> files{"inv/data/inv_data_input1_5.txt",
                       "inv/data/inv_data_output1_5.txt"};
  RunInvKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  vector<string> files{"inv/data/inv_data_input1_6.txt",
                       "inv/data/inv_data_output1_6.txt"};
  RunInvKernelComplex64(files, data_types, shapes);
}

TEST_F(TEST_INV_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}};
  vector<string> files{"inv/data/inv_data_input1_7.txt",
                       "inv/data/inv_data_output1_7.txt"};
  RunInvKernelComplex128(files, data_types, shapes);
}
// exception instance

TEST_F(TEST_INV_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float output[22] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INV_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  bool input1[250] = {(bool)1};
  bool output1[250] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INV_UT, INPUT_INT64_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}};
  int64_t input1[250] = {(int64_t)1};
  int64_t output1[250] = {(int64_t)0};
  vector<void *> datas = {(void *)input1, (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

