#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include <cmath>

#undef private
#undef protected
#include "Eigen/SVD"
#include "Eigen/Core"
#include "Eigen/Dense"

using namespace std;
using namespace aicpu;

class TEST_ISINF_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "IsInf", "IsInf")                 \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]});

template <typename T1, typename T2>
void RunIsInfKernel(vector<string> data_files, vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size] = {T1(0)};
  ;

  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 output[output_size] = {false};
  vector<void *> datas = {(void *)input1, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T2 output_exp[output_size] = {0};

  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

template <typename T1, typename T2>
void RunIsInfKernel_INF(vector<string> data_files, vector<DataType> data_types,
                        vector<vector<int64_t>> &shapes, T2 __ref1, T2 __ref2) {
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size] = {T1(0)};
  for (uint64_t i = 0; i < input1_size; i++) {
    if (i % 2 == 0) {
      input1[i] = *((T1 *)&__ref1);
    } else {
      input1[i] = *((T1 *)&__ref2);
    }
  }
  uint64_t output_size = CalTotalElements(shapes, 1);
  bool output[output_size];
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  string data_path = ktestcaseFilePath + data_files[1];
  bool output_exp[output_size] = {0};
  bool status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ISINF_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ISINF_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  bool output[12] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ISINF_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ISINF_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"is_inf/data/is_inf_data_input_1.txt",
                       "is_inf/data/is_inf_data_output_1.txt"};
  RunIsInfKernel<Eigen::half, bool>(files, data_types, shapes);
}

TEST_F(TEST_ISINF_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"is_inf/data/is_inf_data_input_2.txt",
                       "is_inf/data/is_inf_data_output_2.txt"};
  RunIsInfKernel<float, bool>(files, data_types, shapes);
}

TEST_F(TEST_ISINF_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_BOOL};
  vector<vector<int64_t>> shapes = {{64, 32, 32}, {64, 32, 32}};
  vector<string> files{"is_inf/data/is_inf_data_input_3.txt",
                       "is_inf/data/is_inf_data_output_3.txt"};
  RunIsInfKernel<double, bool>(files, data_types, shapes);
}

TEST_F(TEST_ISINF_UT, INPUT_FILE_DTYPE_FLOAT16_INF_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10}, {10}};
  vector<string> files{"is_inf/data/is_inf_data_input_4.txt",
                       "is_inf/data/is_inf_data_output_4.txt"};
  RunIsInfKernel_INF<Eigen::half, int16_t>(files, data_types, shapes, 0x7C00,
                                           0xFC00);
}

TEST_F(TEST_ISINF_UT, INPUT_FILE_DTYPE_FLOAT_INF_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10}, {10}};
  vector<string> files{"is_inf/data/is_inf_data_input_5.txt",
                       "is_inf/data/is_inf_data_output_5.txt"};
  RunIsInfKernel_INF<float, int32_t>(files, data_types, shapes, 0x7F800000,
                                     0xFF800000);
}

TEST_F(TEST_ISINF_UT, INPUT_FILE_DTYPE_DOUBLE_INF_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_BOOL};
  vector<vector<int64_t>> shapes = {{10}, {10}};
  vector<string> files{"is_inf/data/is_inf_data_input6.txt",
                       "is_inf/data/is_inf_data_output_6.txt"};
  RunIsInfKernel_INF<double, int64_t>(files, data_types, shapes,
                                      0x7FF0000000000000, 0xFFF0000000000000);
}
