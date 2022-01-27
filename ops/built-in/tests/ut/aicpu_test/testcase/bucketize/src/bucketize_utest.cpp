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

class TEST_BUCKETIZE_UT : public testing::Test {};

template <typename T>
void BucketizeCalcExpect(const NodeDef &node_def, T expect_out[],
                         std::vector<float> bound) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  int64_t input0_num = input0->NumElements();
  for (int64_t i = 0; i < input0_num; i++) {
    auto first_bigger_it =
        std::upper_bound(bound.begin(), bound.end(), input0_data[i]);
    expect_out[i] = first_bigger_it - bound.begin();
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas, bound)           \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Bucketize", "Bucketize")         \
      .Input({"x1", (data_types)[0], (shapes)[0], (datas)[0]})     \
      .Output({"y", (data_types)[1], (shapes)[1], (datas)[1]})     \
      .Attr("boundaries", (bound))

// read input and output data from files which generate by your python file
template <typename T1, typename T2>
void RunBucketizeKernel(vector<string> data_files, vector<DataType> data_types,
                        vector<vector<int64_t>> &shapes, vector<float> bound) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 output[output_size];
  vector<void *> datas = {(void *)input1, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, bound);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 output_exp[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

// only generate input data by SetRandomValue,
// and calculate output by youself function
template <typename T1>
void RunBucketizeKernel2(vector<DataType> data_types,
                         vector<vector<int64_t>> &shapes, vector<float> bound) {
  // gen data use SetRandomValue for input1
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  SetRandomValue<T1>(input, input_size);

  uint64_t output_size = CalTotalElements(shapes, 1);
  int32_t *output = new int32_t[output_size];
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, bound);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  int32_t *output_exp = new int32_t[output_size];

  BucketizeCalcExpect<T1>(*node_def.get(), output_exp, bound);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input;
  delete[] output;
  delete[] output_exp;
}

namespace {
std::vector<float> bound = {0, 10, 50, 100};
}
TEST_F(TEST_BUCKETIZE_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 10}, {2, 10}};
  vector<string> files{"bucketize/data/bucketize_data_input1_1.txt",
                       "bucketize/data/bucketize_data_output1_1.txt"};
  RunBucketizeKernel<int32_t, int32_t>(files, data_types, shapes, bound);
}

TEST_F(TEST_BUCKETIZE_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{10, 3}, {10, 3}};
  vector<string> files{"bucketize/data/bucketize_data_input1_2.txt",
                       "bucketize/data/bucketize_data_output1_2.txt"};
  RunBucketizeKernel<int64_t, int32_t>(files, data_types, shapes, bound);
}

TEST_F(TEST_BUCKETIZE_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{15, 5}, {15, 5}};
  vector<string> files{"bucketize/data/bucketize_data_input1_3.txt",
                       "bucketize/data/bucketize_data_output1_3.txt"};
  RunBucketizeKernel<float, int32_t>(files, data_types, shapes, bound);
}

TEST_F(TEST_BUCKETIZE_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32};
  vector<vector<int64_t>> shapes = {{5, 20}, {5, 20}};
  vector<string> files{"bucketize/data/bucketize_data_input1_4.txt",
                       "bucketize/data/bucketize_data_output1_4.txt"};
  RunBucketizeKernel<double, int32_t>(files, data_types, shapes, bound);
}

TEST_F(TEST_BUCKETIZE_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, bound);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {2, 11}};
  int32_t input[22] = {(int32_t)1};
  int64_t output[22] = {(int64_t)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, bound);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  int64_t output[22] = {(int64_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, bound);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BUCKETIZE_UT, DATA_TYPE_INT32_SUCC2) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{128, 1024}, {128, 1024}};
  RunBucketizeKernel2<int32_t>(data_types, shapes, bound);
}
