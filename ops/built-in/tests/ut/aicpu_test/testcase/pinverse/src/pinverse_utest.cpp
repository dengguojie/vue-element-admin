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

class TEST_PINVERSE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, rcond)           \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Pinverse", "Pinverse")           \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})           \
      .Attr("rcond", rcond)                                 

// read input and output data from files which generate by your python file
template<typename T>
void RunPinverseKernel(vector<string> data_files,
               vector<DataType> data_types,
               vector<vector<int64_t>> &shapes, float rcond) {
  // read data from file for input
  std::string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T *input = new T[input_size];
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, rcond);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T *output_exp = new T[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_PINVERSE_UT, Pinverse_x_SUCCESS_float) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 5}, {5, 3}};
  vector<string> files = {"pinverse/data/pinverse_data_input_float.txt",
                          "pinverse/data/pinverse_data_output_float.txt"};
  RunPinverseKernel<float>(files, data_types, shapes, 1e-15);
}

TEST_F(TEST_PINVERSE_UT, Pinverse_x_SUCCESS_double) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}};
  vector<string> files = {"pinverse/data/pinverse_data_input_double.txt",
                          "pinverse/data/pinverse_data_output_double.txt"};
  RunPinverseKernel<double>(files, data_types, shapes, 1e-15);
}

TEST_F(TEST_PINVERSE_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  bool input[6] = {(bool)1};
  bool output[6] = {(bool)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 1e-15);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_PINVERSE_UT, Pinverse_rcond_SUCCESS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}};
  vector<string> files = {"pinverse/data/pinverse_data_input_double_rcond.txt",
                          "pinverse/data/pinverse_data_output_double_rcond.txt"};
  float rcond = 1e-8;
  RunPinverseKernel<double>(files, data_types, shapes, rcond);
}

TEST_F(TEST_PINVERSE_UT, Pinverse_rcond_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}};
  vector<string> files = {"pinverse/data/pinverse_data_input_double_rcond.txt",
                          "pinverse/data/pinverse_data_output_double_rcond.txt"};
  double rcond = 0.1;
  RunPinverseKernel<double>(files, data_types, shapes, rcond);
}

TEST_F(TEST_PINVERSE_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3, 3}, {3, 4, 2}};
  float input[27] = {(float)1};
  float output[24] = {(float)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
