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

class TEST_IFFT_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "IFFT", "IFFT")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})

template <typename T1, typename T2>
bool ReadFileComplexIFFT(std::string file_name, T1 output[],
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

      T2 temp;
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

// read input and output data from files which generate by your python file
template <typename T1, typename T2>
void RunIFFTKernel(vector<string> data_files, vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 input[input_size];
  bool status = ReadFileComplexIFFT<T1, T2>(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T1 output[output_size];
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T1 output_exp[output_size];
  status = ReadFileComplexIFFT<T1, T2>(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_IFFT_UT, DATA_TYPE_COMPLEX64_2D_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{4, 32}, {4, 32}};
  vector<string> files{"ifft/data/ifft_data_input1_1.txt",
                       "ifft/data/ifft_data_output1_1.txt"};
  RunIFFTKernel<std::complex<float>, float>(files, data_types, shapes);
}

TEST_F(TEST_IFFT_UT, DATA_TYPE_COMPLEX64_3D_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{8, 32, 32}, {8, 32, 32}};
  vector<string> files{"ifft/data/ifft_data_input1_2.txt",
                       "ifft/data/ifft_data_output1_2.txt"};
  RunIFFTKernel<std::complex<float>, float>(files, data_types, shapes);
}

TEST_F(TEST_IFFT_UT, DATA_TYPE_COMPLEX128_3D_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{32, 32, 32}, {32, 32, 32}};
  vector<string> files{"ifft/data/ifft_data_input1_3.txt",
                       "ifft/data/ifft_data_output1_3.txt"};
  RunIFFTKernel<std::complex<double>, double>(files, data_types, shapes);
}

// exception
TEST_F(TEST_IFFT_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 3, 3}, {4, 3, 3}};
  int32_t input[36] = {(int32_t)1};
  int32_t output[36] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IFFT_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {4}};
  int32_t input[4] = {(int32_t)1};
  int32_t output[4] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IFFT_UT, INPUT_TYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 3, 3}, {4, 3, 3}};
  int32_t input[36] = {(int32_t)1};
  int32_t output[36] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}