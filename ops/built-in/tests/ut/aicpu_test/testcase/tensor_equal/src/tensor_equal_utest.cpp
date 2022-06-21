#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include "utils/kernel_util.h"

#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_TENSOR_EQUAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "TensorEqual", "TensorEqual")     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunTensorEqualKernel(vector<string> data_files,
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

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 1, 3}, {2, 1, 3}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_1.txt",
                       "tensor_equal/data/tensor_equal_data_input2_1.txt",
                       "tensor_equal/data/tensor_equal_data_output1_1.txt"};
  RunTensorEqualKernel<int32_t, int32_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1024, 8}, {1024, 8}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_2.txt",
                       "tensor_equal/data/tensor_equal_data_input2_2.txt",
                       "tensor_equal/data/tensor_equal_data_output1_2.txt"};
  RunTensorEqualKernel<int64_t, int64_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{4, 1024}, {4, 1024}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_3.txt",
                       "tensor_equal/data/tensor_equal_data_input2_3.txt",
                       "tensor_equal/data/tensor_equal_data_output1_3.txt"};
  RunTensorEqualKernel<float, float, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1, 3}, {1, 3}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_4.txt",
                       "tensor_equal/data/tensor_equal_data_input2_4.txt",
                       "tensor_equal/data/tensor_equal_data_output1_4.txt"};
  RunTensorEqualKernel<double, double, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_BOOL};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_5.txt",
                       "tensor_equal/data/tensor_equal_data_input2_5.txt",
                       "tensor_equal/data/tensor_equal_data_output1_5.txt"};
  RunTensorEqualKernel<int8_t, int8_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_BOOL};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_6.txt",
                       "tensor_equal/data/tensor_equal_data_input2_6.txt",
                       "tensor_equal/data/tensor_equal_data_output1_6.txt"};
  RunTensorEqualKernel<uint8_t, uint8_t, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 2}, {2, 2, 2}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_7.txt",
                       "tensor_equal/data/tensor_equal_data_input2_7.txt",
                       "tensor_equal/data/tensor_equal_data_output1_7.txt"};
  RunTensorEqualKernel<Eigen::half, Eigen::half, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_DOUBLE_FIXED_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3, 2}, {2, 3, 2}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_8.txt",
                       "tensor_equal/data/tensor_equal_data_input2_8.txt",
                       "tensor_equal/data/tensor_equal_data_output1_8.txt"};
  RunTensorEqualKernel<double, double, bool>(files, data_types, shapes);
}

TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_FLOAT16_FIXED_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3, 2}, {2, 3, 2}, {1}};
  vector<string> files{"tensor_equal/data/tensor_equal_data_input1_9.txt",
                       "tensor_equal/data/tensor_equal_data_input2_9.txt",
                       "tensor_equal/data/tensor_equal_data_output1_9.txt"};
  RunTensorEqualKernel<Eigen::half, Eigen::half, bool>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_TENSOR_EQUAL_UT, DATA_TYPE_INT32_DIFF_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2}, {1}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[4] = {(int32_t)1};
  bool output[1] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool output_exp[1] = {false};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_TENSOR_EQUAL_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {1}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  bool output[1] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TENSOR_EQUAL_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {1}};
  bool output[1] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TENSOR_EQUAL_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {1}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[1] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}