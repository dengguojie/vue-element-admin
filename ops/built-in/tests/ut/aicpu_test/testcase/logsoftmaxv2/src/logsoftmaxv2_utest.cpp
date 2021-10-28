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

class TEST_LOGSOFTMAXV2_UT : public testing::Test {};
#define CREATE_NODEDEF(shapes, data_types, datas, axes)            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "LogSoftmaxV2", "LogSoftmaxV2")   \
      .Input({"logits", data_types[0], shapes[0], datas[0]})       \
      .Output({"logsoftmax", data_types[1], shapes[1], datas[1]})  \
      .Attr("axes", axes)

// read input and output data from files which generate by your python file
template <typename T1, typename T2>
void RunLogSoftmaxV2Kernel(vector<string> data_files,
                           vector<DataType> data_types,
                           vector<vector<int64_t>> &shapes,
                           vector<int64_t> axes) {
  // read data from file for input1
  string data_path_1 = ktestcaseFilePath + data_files[0];
  uint64_t logits_size = CalTotalElements(shapes, 0);
  T1 *logits = new T1[logits_size];
  bool status = ReadFile(data_path_1, logits, logits_size);
  EXPECT_EQ(status, true);

  uint64_t logsoftmax_size = CalTotalElements(shapes, 1);
  T2 *logsoftmax = new T2[logsoftmax_size];
  vector<void *> datas = {(void *)logits, (void *)logsoftmax};

  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  string data_path_2 = ktestcaseFilePath + data_files[1];
  T2 *logsoftmax_exp = new T2[logsoftmax_size];
  status = ReadFile(data_path_2, logsoftmax_exp, logsoftmax_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(logsoftmax, logsoftmax_exp, logsoftmax_size);
  EXPECT_EQ(compare, true);
  delete[] logits;
  delete[] logsoftmax;
  delete[] logsoftmax_exp;
}

TEST_F(TEST_LOGSOFTMAXV2_UT, DATA_TYPE_FLOAT16) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12, 10}, {12, 10}};
  vector<string> files{
      "logsoftmaxv2/data/logsoftmaxv2_data_input_float16.txt",
      "logsoftmaxv2/data/logsoftmaxv2_data_output_float16.txt"};
  vector<int64_t> axes = {-1};
  RunLogSoftmaxV2Kernel<Eigen::half, Eigen::half>(files, data_types, shapes,
                                                  axes);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, DATA_TYPE_FLOAT) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{50, 5, 10, 10}, {50, 5, 10, 10}};
  vector<string> files{"logsoftmaxv2/data/logsoftmaxv2_data_input_float.txt",
                       "logsoftmaxv2/data/logsoftmaxv2_data_output_float.txt"};
  vector<int64_t> axes = {2};
  RunLogSoftmaxV2Kernel<float, float>(files, data_types, shapes, axes);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, DATA_TYPE_DOUBLE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{5, 20, 3, 6, 4}, {5, 20, 3, 6, 4}};
  vector<string> files{"logsoftmaxv2/data/logsoftmaxv2_data_input_double.txt",
                       "logsoftmaxv2/data/logsoftmaxv2_data_output_double.txt"};
  vector<int64_t> axes = {2};
  RunLogSoftmaxV2Kernel<double, double>(files, data_types, shapes, axes);
}
// exception instance
TEST_F(TEST_LOGSOFTMAXV2_UT, AXES_NUM_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
  int32_t input[120] = {(int32_t)1};
  int32_t output[120] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  vector<int64_t> axes = {3};
  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, AXES_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
  int32_t input[120] = {(int32_t)1};
  int32_t output[120] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  vector<int64_t> axes = {5, 3};
  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{}, {}};
  double input[6] = {(double)1.0};
  double output[6] = {(double)0.0};
  vector<void *> datas = {(void *)input, (void *)output};
  vector<int64_t> axes = {1};
  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_STRING, DT_STRING};
  vector<vector<int64_t>> shapes = {{4, 5, 6}, {4, 5, 6}};
  int32_t input[120] = {'1'};
  int32_t output[120] = {'0'};
  vector<void *> datas = {(void *)input, (void *)output};
  vector<int64_t> axes = {1};
  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  double output[60] = {(double)0.0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  vector<int64_t> axes = {1};
  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_LOGSOFTMAXV2_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}};
  bool input[60] = {(bool)1};
  bool output[60] = {(bool)0};
  vector<void *> datas = {(void *)input, (void *)output};
  vector<int64_t> axes = {1};
  CREATE_NODEDEF(shapes, data_types, datas, axes);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}