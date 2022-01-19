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

class TEST_DIAG_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Diag", "Diag")                   \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})      \
      .Output({"y", (data_types)[1], (shapes)[1], (datas)[1]})

TEST_F(TEST_DIAG_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {36}};
  int32_t input[6] = {(int32_t)1};
  int32_t output[36] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIAG_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT8, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11, 2, 11}};
  int8_t input[22] = {(int8_t)1};
  int32_t output[22 * 22] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIAG_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11, 2, 11}};
  int32_t output[22] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIAG_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11, 2, 11}};
  bool input[22] = {static_cast<bool>(1)};
  int32_t output[22 * 22] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_INT32_SUCC_1D) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3}, {3, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_1.txt";
  constexpr uint64_t input_size = 3;
  int32_t input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_1.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_INT32_SUCC_2D) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 6}, {3, 6, 3, 6}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_2.txt";
  constexpr uint64_t input_size = 3 * 6;
  int32_t input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_2.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_INT32_SUCC_3D) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 5, 3}, {2, 5, 3, 2, 5, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_3.txt";
  constexpr uint64_t input_size = 2 * 5 * 3;
  int32_t input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_3.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{5, 3, 7}, {5, 3, 7, 5, 3, 7}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_4.txt";
  constexpr uint64_t input_size = 5 * 3 * 7;
  Eigen::half input[input_size] = {(Eigen::half)0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  Eigen::half output[output_size] = {(Eigen::half)0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_4.txt";
  Eigen::half output_exp[output_size] = {(Eigen::half)0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 3, 7}, {5, 3, 7, 5, 3, 7}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_5.txt";
  constexpr uint64_t input_size = 5 * 3 * 7;
  float input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  float output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_5.txt";
  float output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{8, 8, 10}, {8, 8, 10, 8, 8, 10}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_6.txt";
  constexpr uint64_t input_size = 8 * 8 * 10;
  double input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  double output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_6.txt";
  double output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{5, 3, 7}, {5, 3, 7, 5, 3, 7}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_7.txt";
  constexpr uint64_t input_size = 5 * 3 * 7;
  int64_t input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  int64_t output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_7.txt";
  int64_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{5, 3, 7}, {5, 3, 7, 5, 3, 7}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_8.txt";
  constexpr uint64_t input_size = 5 * 3 * 7;
  std::complex<float> input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  std::complex<float> output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_8.txt";
  std::complex<float> output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIAG_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{5, 3, 7}, {5, 3, 7, 5, 3, 7}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "diag/data/diag_data_input_9.txt";
  constexpr uint64_t input_size = 5 * 3 * 7;
  std::complex<double> input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = input_size * input_size;
  std::complex<double> output[output_size] = {0};
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "diag/data/diag_data_output_9.txt";
  std::complex<double> output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}