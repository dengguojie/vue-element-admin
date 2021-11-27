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

class TEST_INVGRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "InvGrad", "InvGrad")             \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunInvGradKernel(vector<string> data_files,
                   vector<DataType> data_types,
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
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT_SUCC_1D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15}, {15}, {15}};
  vector<string> files{"invgrad/data/invgrad_data_input1_1.txt",
                       "invgrad/data/invgrad_data_input2_1.txt",
                       "invgrad/data/invgrad_data_output1_1.txt"};
  RunInvGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT16_SUCC_1D) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{16,12}, {16,12}, {16,12}};
  vector<string> files{"invgrad/data/invgrad_data_input1_2.txt",
                       "invgrad/data/invgrad_data_input2_2.txt",
                       "invgrad/data/invgrad_data_output1_2.txt"};
  RunInvGradKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1, 128}, {1, 128}, {1, 128}};
  vector<string> files{"invgrad/data/invgrad_data_input1_3.txt",
                       "invgrad/data/invgrad_data_input2_3.txt",
                       "invgrad/data/invgrad_data_output1_3.txt"};
  RunInvGradKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{10, 5, 5}, {10, 5, 5}, {10, 5, 5}};
  vector<string> files{"invgrad/data/invgrad_data_input1_4.txt",
                       "invgrad/data/invgrad_data_input2_4.txt",
                       "invgrad/data/invgrad_data_output1_4.txt"};
                      
   RunInvGradKernel< std::complex<float>,  std::complex<float>,  std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}, {6, 6, 6}};
  vector<string> files{"invgrad/data/invgrad_data_input1_5.txt",
                       "invgrad/data/invgrad_data_input2_5.txt",
                       "invgrad/data/invgrad_data_output1_5.txt"};
  RunInvGradKernel< std::complex<double>,  std::complex<double>,  std::complex<double>>(files, data_types, shapes);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT16_SUCC_ADD1) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{7,1024}, {7,1024}, {7,1024}};
  vector<string> files{"invgrad/data/invgrad_data_input1_6.txt",
                       "invgrad/data/invgrad_data_input2_6.txt",
                       "invgrad/data/invgrad_data_output1_6.txt"};
  RunInvGradKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT16_SUCC_ADD2) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1,1024}, {1,1024}, {1,1024}};
  vector<string> files{"invgrad/data/invgrad_data_input1_7.txt",
                       "invgrad/data/invgrad_data_input2_7.txt",
                       "invgrad/data/invgrad_data_output1_7.txt"};
  RunInvGradKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}


TEST_F(TEST_INVGRAD_UT, DATA_TYPE_FLOAT16_SUCC_ADD3) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{8192,1}, {8192,1}, {8192,1}};
  vector<string> files{"invgrad/data/invgrad_data_input1_8.txt",
                       "invgrad/data/invgrad_data_input2_8.txt",
                       "invgrad/data/invgrad_data_output1_8.txt"};
  RunInvGradKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}


// exception instance
TEST_F(TEST_INVGRAD_UT, INPUT_NUMBER_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  float input1[12] = {(float)1};
  float input2[16] = {(float)0};
  float output[16] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  float input1[22] = {(float)1};
  Eigen::half input2[22] = {(Eigen::half)0};
  float output[22] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  float output[22] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INVGRAD_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
