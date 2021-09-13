#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_ELUGRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "EluGrad", "EluGrad")             \
      .Input({"grads", data_types[0], shapes[0], datas[0]})        \
      .Input({"activations", data_types[1], shapes[1], datas[1]})  \
      .Output({"y", data_types[2], shapes[2], datas[2]})

template <typename T1, typename T2, typename T3>
void RunEluGradKernel(vector<string> data_files, vector<DataType> data_types,
                      vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

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

TEST_F(TEST_ELUGRAD_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12, 15, 30}, {12, 15, 30}, {12, 15, 30}};
  vector<string> files{"elugrad/data/elugrad_data_input1_1.txt",
                       "elugrad/data/elugrad_data_input1_2.txt",
                       "elugrad/data/elugrad_data_output_1.txt"};
  RunEluGradKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types,
                                                          shapes);
}

TEST_F(TEST_ELUGRAD_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{12, 15, 30}, {12, 15, 30}, {12, 15, 30}};
  vector<string> files{"elugrad/data/elugrad_data_input2_1.txt",
                       "elugrad/data/elugrad_data_input2_2.txt",
                       "elugrad/data/elugrad_data_output_2.txt"};
  RunEluGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_ELUGRAD_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{12, 15, 30}, {12, 15, 30}, {12, 15, 30}};
  vector<string> files{"elugrad/data/elugrad_data_input3_1.txt",
                       "elugrad/data/elugrad_data_input3_2.txt",
                       "elugrad/data/elugrad_data_output_3.txt"};
  RunEluGradKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_ELUGRAD_UT, DATA_TYPE_FLOAT16_ELUGRAD_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {12, 15, 300}, {12, 15, 300}, {12, 15, 300}};
  vector<string> files{"elugrad/data/elugrad_data_input4_1.txt",
                       "elugrad/data/elugrad_data_input4_2.txt",
                       "elugrad/data/elugrad_data_output_4.txt"};
  RunEluGradKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types,
                                                          shapes);
}

TEST_F(TEST_ELUGRAD_UT, DATA_TYPE_FLOAT_ELUGRAD_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {
      {12, 15, 300}, {12, 15, 300}, {12, 15, 300}};
  vector<string> files{"elugrad/data/elugrad_data_input5_1.txt",
                       "elugrad/data/elugrad_data_input5_2.txt",
                       "elugrad/data/elugrad_data_output_5.txt"};
  RunEluGradKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_ELUGRAD_UT, DATA_TYPE_DOUBLE_ELUGRAD_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {
      {12, 15, 300}, {12, 15, 300}, {12, 15, 300}};
  vector<string> files{"elugrad/data/elugrad_data_input6_1.txt",
                       "elugrad/data/elugrad_data_input6_2.txt",
                       "elugrad/data/elugrad_data_output_6.txt"};
  RunEluGradKernel<double, double, double>(files, data_types, shapes);
}
TEST_F(TEST_ELUGRAD_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[16] = {(int32_t)0};
  bool output[16] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ELUGRAD_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ELUGRAD_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ELUGRAD_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
