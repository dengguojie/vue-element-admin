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

class TEST_COMBINATIONS_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, r, with_replacement) \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  NodeDefBuilder(node_def.get(), "Combinations", "Combinations")       \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("r", r)                                                    \
      .Attr("with_replacement", with_replacement)

template <typename T>
void RunCombinationsKernel(vector<string> data_files, vector<DataType> data_types, vector<vector<int64_t>>& shapes,
                           int r = 2, bool with_replacement = false) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T* input = new T[input_size];
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T* output = new T[output_size];
  vector<void*> datas = {(void*)input, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas, r, with_replacement);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T* output_exp = new T[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);

  delete[] input;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_1.txt",
                       "combinations/data/combinations_data_output1_1.txt"};
  RunCombinationsKernel<int32_t>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{6}, {15, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_2.txt",
                       "combinations/data/combinations_data_output1_2.txt"};
  RunCombinationsKernel<int64_t>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_3.txt",
                       "combinations/data/combinations_data_output1_3.txt"};
  RunCombinationsKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_BOOL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{8}, {28, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_4.txt",
                       "combinations/data/combinations_data_output1_4.txt"};
  RunCombinationsKernel<bool>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_5.txt",
                       "combinations/data/combinations_data_output1_5.txt"};
  RunCombinationsKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_6.txt",
                       "combinations/data/combinations_data_output1_6.txt"};
  RunCombinationsKernel<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_7.txt",
                       "combinations/data/combinations_data_output1_7.txt"};
  RunCombinationsKernel<int16_t>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_8.txt",
                       "combinations/data/combinations_data_output1_8.txt"};
  RunCombinationsKernel<int8_t>(files, data_types, shapes);
}

TEST_F(TEST_COMBINATIONS_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_9.txt",
                       "combinations/data/combinations_data_output1_9.txt"};
  RunCombinationsKernel<uint8_t>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_COMBINATIONS_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 4}};
  int32_t input[16] = {(int32_t)1};
  int32_t output[16] = {(int32_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas, 2, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMBINATIONS_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  int32_t input[4] = {(int32_t)1};
  int64_t output[12] = {(int64_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas, 2, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMBINATIONS_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  bool output[12] = {(bool)0};
  vector<void*> datas = {(void*)nullptr, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas, 2, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMBINATIONS_UT, ATTR_R_EXCEPTION1) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  int32_t input[4] = {(int32_t)1};
  int32_t output[12] = {(int32_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas, -1, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMBINATIONS_UT, ATTR_R_EXCEPTION2) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {6, 2}};
  int32_t input[4] = {(int32_t)1};
  int32_t output[12] = {(int32_t)0};
  vector<void*> datas = {(void*)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas, 5, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_COMBINATIONS_UT, ATTR_R_TEST) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{6}, {20, 3}};
  vector<string> files{"combinations/data/combinations_data_input1_10.txt",
                       "combinations/data/combinations_data_output1_10.txt"};
  RunCombinationsKernel<int32_t>(files, data_types, shapes, 3);
}

TEST_F(TEST_COMBINATIONS_UT, ATTR_WITHREPLACEMENT_TEST) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {10, 2}};
  vector<string> files{"combinations/data/combinations_data_input1_11.txt",
                       "combinations/data/combinations_data_output1_11.txt"};
  RunCombinationsKernel<int32_t>(files, data_types, shapes, 2, true);
}
