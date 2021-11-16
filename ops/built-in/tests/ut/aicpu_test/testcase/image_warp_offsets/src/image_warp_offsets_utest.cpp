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

class TEST_IMAGE_WARP_OFFSETS_UT : public testing::Test {};

template <typename T>
void CalcExpectWithSameShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num == input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = input0_data[j] < input1_data[j] ? true : false;
    }
  }
}

template <typename T>
void CalcExpectWithDiffShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num > input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      int64_t i = j % input1_num;
      expect_out[j] = input0_data[j] < input1_data[i] ? true : false;
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "IMGWarpOffsets", "IMGWarpOffsets") \
      .Input({"x1", data_types[0], shapes[0], datas[0]})             \
      .Input({"x2", data_types[1], shapes[1], datas[1]})             \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunIMGWarpOffsetsKernel(vector<string> data_files,
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
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
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

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, NO_KERNEL_FAILED) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 3}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT0_RANK_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 2}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT0_SHPAE_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, -1}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT0_DIM_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2, 3}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT1_RANK_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2}, {1, 4, 2, 2, -1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT1_SHPAE_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 2}, {1, 2, 2, -1}, {1, 4, 2, 2, -1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INPUT1_DIM_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 3}, {1, 2, 2, 2}, {1, 4, 2, 2, 2}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, OUTPUT_RANK_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, OUTPUT_SHPAE_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 2}, {1, 2, 2, 2}, {1, 4, 2, 2, -1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, SHPAE_MISMATCH_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 1}};
  int32_t data[40] = {0};
  vector<void *> datas = {(void *)data, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IMAGE_WARP_OFFSETS_UT, INVALID_VALUE_FAILED) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {
      {1, 2, 2, 3}, {1, 4, 2, 2}, {1, 4, 2, 2, 3}};
  Eigen::half data1[12] = {Eigen::half(1.1)};
  int32_t data[40] = {30, 30, 30, 30, 30, 30, 20, 20, 20};
  vector<void *> datas = {(void *)data1, (void *)data, (void *)data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
