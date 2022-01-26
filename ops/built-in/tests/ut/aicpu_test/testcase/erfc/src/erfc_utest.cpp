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

class TEST_ERFC_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Erfc", "Erfc")                   \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})      \
      .Output({"y", (data_types)[1], (shapes)[1], (datas)[1]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2>
void RunErfcKernel(vector<string> data_files, vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path_1 = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  bool status = ReadFile(data_path_1, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  string data_path_2 = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path_2, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_ERFC_UT, DATA_TYPE_FlOAT16) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  vector<string> files{"erfc/data/erfc_data_input_float16.txt",
                       "erfc/data/erfc_data_output_float16.txt"};
  RunErfcKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_FlOAT16_BIG) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{4, 50, 8, 10}, {4, 50, 8, 10}};
  vector<string> files{"erfc/data/erfc_data_input_float16_big.txt",
                       "erfc/data/erfc_data_output_float16_big.txt"};
  RunErfcKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_FlOAT) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  vector<string> files{"erfc/data/erfc_data_input_float.txt",
                       "erfc/data/erfc_data_output_float.txt"};
  RunErfcKernel<float, float>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_FlOAT_BIG) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 50, 8, 10}, {4, 50, 8, 10}};
  vector<string> files{"erfc/data/erfc_data_input_float_big.txt",
                       "erfc/data/erfc_data_output_float_big.txt"};
  RunErfcKernel<float, float>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_DOUBLE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  vector<string> files{"erfc/data/erfc_data_input_double.txt",
                       "erfc/data/erfc_data_output_double.txt"};
  RunErfcKernel<double, double>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_DOUBL_BIG) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4, 50, 8, 10}, {4, 50, 8, 10}};
  vector<string> files{"erfc/data/erfc_data_input_double_big.txt",
                       "erfc/data/erfc_data_output_double_big.txt"};
  RunErfcKernel<double, double>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_FLOAT_1) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  vector<string> files{"erfc/data/erfc_data_input_float_1.txt",
                       "erfc/data/erfc_data_output_float_1.txt"};
  RunErfcKernel<float, float>(files, data_types, shapes);
}
TEST_F(TEST_ERFC_UT, DATA_TYPE_DOUBL_1) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  vector<string> files{"erfc/data/erfc_data_input_double_1.txt",
                       "erfc/data/erfc_data_output_double_1.txt"};
  RunErfcKernel<double, double>(files, data_types, shapes);
}
// exception instance
TEST_F(TEST_ERFC_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{}, {}};
  double input[6] = {(double)1};
  double output[6] = {(double)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ERFC_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
  int32_t input[120] = {1};
  int32_t output[120] = {0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ERFC_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  double output[60] = {(double)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ERFC_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  bool input[60] = {(bool)1};
  bool output[60] = {(bool)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}