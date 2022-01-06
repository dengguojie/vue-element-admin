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

class TEST_SIGMOIDGRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "SigmoidGrad", "SigmoidGrad")     \
      .Input({"y", data_types[0], shapes[0], datas[0]})            \
      .Input({"dy", data_types[1], shapes[1], datas[1]})           \
      .Output({"z", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_SIGMOIDGRAD_UT, DOUBLE_INPUT_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 12}, {3, 12}, {3, 12}};

  // read data from file for input1
  string data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_input1_1.txt";
  double input1[36] = {0};
  bool status = ReadFile(data_path, input1, 36);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_input2_1.txt";
  double input2[36] = {0};
  status = ReadFile(data_path, input2, 36);
  EXPECT_EQ(status, true);

  double output[36] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_output1_1.txt";
  double output_exp[36] = {0};
  status = ReadFile(data_path, output_exp, 36);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, 36);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, DOUBLE_SAME_INPUT_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{32, 32, 32}, {32, 32, 32}, {32, 32, 32}};

  // read data from file for input1
  string data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_input1_2.txt";
  constexpr uint64_t input1_size = 32 * 32 * 32;
  double input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_input2_2.txt";
  constexpr uint64_t input2_size = 32 * 32 * 32;
  double input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 32 * 32 * 32;
  double output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_output1_2.txt";
  double output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, FLOAT_SAME_INPUT_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}, {15, 12, 30}};

  // read data from file for input1
  string data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_input1_3.txt";
  constexpr uint64_t input1_size = 15 * 12 * 30;
  float input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_input2_3.txt";
  constexpr uint64_t input2_size = 15 * 12 * 30;
  float input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 15 * 12 * 30;
  float output[output_size] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path =
      ktestcaseFilePath + "sigmoid_grad/data/sigmoid_grad_data_output1_3.txt";
  float output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, FLOAT16_SCALAR_SIGMOIDGRAD_SCALAR_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};

  Eigen::half input1[2] = {(Eigen::half)2.0, (Eigen::half)5.0};
  Eigen::half input2[2] = {(Eigen::half)2.0, (Eigen::half)5.0};
  Eigen::half output[2] = {(Eigen::half)0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  Eigen::half output_exp[2] = {(Eigen::half)-4.0, (Eigen::half)-100.0};
  bool compare = CompareResult(output, output_exp, 2);
}

TEST_F(TEST_SIGMOIDGRAD_UT, COMPLEX64_SCALAR_SIGMOIDGRAD_VECTOR_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};

  std::complex<float> input1[2] = {2.0, 2.0};
  std::complex<float> input2[2] = {2.0, 3.0};
  std::complex<float> output[2] = {0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::complex<float> output_exp[2] = {-4.0, -6.0};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, COMPLEX128_VECTOR_SIGMOIDGRAD_SCALAR_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};

  std::complex<double> input1[2] = {2.0, 5.0};
  std::complex<double> input2[2] = {3.0, 5.0};
  std::complex<double> output[2] = {0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::complex<double> output_exp[2] = {-6.0, -100.0};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, COMPLEX128_VECTOR_SIGMOIDGRAD_VECTOR_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 1, 1}, {2, 1, 1}, {2, 1, 1}};

  std::complex<double> input1[2] = {2.0, 5.0};
  std::complex<double> input2[2] = {3.0, 5.0};
  std::complex<double> output[2] = {0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::complex<double> output_exp[2] = {-6.0, -100.0};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, FLOAT_VECTOR_SIGMOIDGRAD_SCALAR_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};

  float input1[2] = {2.0F, 5.0F};
  float input2[2] = {3.0F, 5.0F};
  float output[2] = {0.0F};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  float output_exp[2] = {-6.0F, -100.0F};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, FLOAT16_VECTOR_SIGMOIDGRAD_VECTOR) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 1, 1, 1}, {2, 1, 1, 1}, {2, 1, 1, 1}};

  Eigen::half input1[2] = {(Eigen::half)2.0, (Eigen::half)3.0};
  Eigen::half input2[2] = {(Eigen::half)3.0, (Eigen::half)2.0};
  Eigen::half output[2] = {(Eigen::half)0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  Eigen::half output_exp[2] = {(Eigen::half)-6.0, (Eigen::half)-12.0};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, COMPLEX64_VECTOR_SIGMOIDGRAD_VECTOR_BOTH) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{1, 2}, {1, 2}, {1, 2}};

  std::complex<float> input1[2] = {2.0, 3.0};
  std::complex<float> input2[2] = {3.0, 2.0};
  std::complex<float> output[2] = {0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  std::complex<float> output_exp[2] = {-6.0, -12.0};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SIGMOIDGRAD_UT, UNDEFINED_DTYPE_EXCEPTION) {
  vector<vector<int64_t>> shapes = {{}, {}, {}};
  vector<DataType> data_types = {DT_UNDEFINED, DT_UNDEFINED, DT_UNDEFINED};
  vector<int32_t> input1 = {1};
  vector<int32_t> input2 = {1};
  vector<int32_t> output = {0};
  vector<void *> datas = {(void *)input1.data(), (void *)input2.data(),
                          (void *)output.data()};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_UT, DTYPE_UNSORPORT) {
  vector<vector<int64_t>> shapes = {{}, {}, {}};
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  bool input1[1] = {false};
  bool input2[1] = {true};
  bool output[1] = {false};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_UT, INPUT_DTYPE_DISMATCH) {
  vector<vector<int64_t>> shapes = {{}, {}, {}};
  vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
  float input1[1] = {1.0F};
  double input2[1] = {2.0};
  double output[1] = {0.0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_UT, NULL_INPUT) {
  vector<vector<int64_t>> shapes = {{1}, {1}, {1, 1}};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  float input1[1] = {1.0F};
  float input2[1] = {2.0F};
  float output[1] = {0.0F};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_UT, OUTPUT_SHAPE_DISMATCH) {
  vector<vector<int64_t>> shapes = {{1, 2}, {1, 2}, {1, 1}};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  float input1[2] = {1.0F};
  float input2[3] = {2.0F};
  float output[1] = {0.0F};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SIGMOIDGRAD_UT, INPUT_SHAPE_DISMATCH) {
  vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {1, 2}};
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  float input1[2] = {1.0F};
  float input2[3] = {2.0F};
  float output[1] = {0.0F};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}