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

Eigen::half half_zero = Eigen::half(0);

class TEST_STN_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, format, stn_ori_channel, use_default_theta, default_theta)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "SpatialTransformer", "SpatialTransformer")                     \
      .Input({"x", data_types[0], shapes[0], datas[0], format})           \
      .Input({"theta", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2], format})            \
      .Attr("stn_ori_channel", stn_ori_channel)                    \
      .Attr("use_default_theta", use_default_theta)               \
      .Attr("default_theta", default_theta);

TEST_F(TEST_STN_UT, 4D_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_FLOAT16, DT_INT8};
  vector<vector<int64_t>> shapes = {{1, 1, 2, 3}, {2}, {1, 1, 2, 3}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "SpatialTransformer/data/SpatialTransformer_data_input1_0.txt";
  constexpr uint64_t input1_size = 1 * 1 * 2 * 3;
  int8_t input1[input1_size] = {-39, -47, -37, 4, -70, -47};

  // read data from file for input2
  data_path = ktestcaseFilePath + "SpatialTransformer/data/SpatialTransformer_data_input2_0.txt";
  constexpr uint64_t input2_size = 2;

  Eigen::half input2[input2_size] = {Eigen::half(-1), Eigen::half(-2)};

  constexpr uint64_t output_size = 1 * 1 * 2 * 3;
  int8_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};
  
  vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
  vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
  CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "SpatialTransformer/data/SpatialTransformer_data_output1_0.txt";
  int8_t output_exp[output_size] = {-35, 0, 0, 0, -34, -32};

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_STN_UT, 5D_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT8, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 1, 2, 3, 16}, {2}, {1, 1, 2, 3, 16}};

  // read data from file for input1
  constexpr uint64_t input1_size = 1 * 1 * 2 * 3 * 16;
  Eigen::half input1[input1_size] = {half_zero};
  for (uint32_t i = 0; i < 16; i ++) {
    if (i < 10) {
      input1[0 * 16 + i] = Eigen::half(-12.304688);
      input1[1 * 16 + i] = Eigen::half(-24.906250);
      input1[2 * 16 + i] = Eigen::half(4);
      input1[3 * 16 + i] = Eigen::half(-42.25);
      input1[4 * 16 + i] = Eigen::half(-35.125);
      input1[5 * 16 + i] = Eigen::half(-70);
    }
  }

  // read data from file for input2
  constexpr uint64_t input2_size = 2;
  int8_t input2[input2_size] = {-2, -1};

  constexpr uint64_t output_size = 1 * 1 * 2 * 3 * 16;
  Eigen::half output[output_size] = {half_zero};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};
  
  vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
  vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
  CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NC1HWC0, 10, use_default_theta, default_theta);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  Eigen::half output_exp[output_size] = {half_zero};
  for (uint32_t i = 0; i < 16; i ++) {
    if (i < 10) {
      output_exp[0 * 16 + i] = Eigen::half(0);
      output_exp[1 * 16 + i] = Eigen::half(-34.0);
      output_exp[2 * 16 + i] = Eigen::half(0);
      output_exp[3 * 16 + i] = Eigen::half(0);
      output_exp[4 * 16 + i] = Eigen::half(-12.302080);
      output_exp[5 * 16 + i] = Eigen::half(-15.499998);
    }
  }

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_STN_UT, 5D_C1_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 1, 2, 3, 16}, {2}, {1, 1, 2, 3, 16}};

  // read data from file for input1
  constexpr uint64_t input1_size = 1 * 1 * 2 * 3 * 16;
  Eigen::half input1[input1_size] = {half_zero};
  for (uint32_t i = 0; i < 1; i ++) {
    if (i < 1) {
      input1[0 * 16 + i] = Eigen::half(-12.304688);
      input1[1 * 16 + i] = Eigen::half(-24.906250);
      input1[2 * 16 + i] = Eigen::half(4);
      input1[3 * 16 + i] = Eigen::half(-42.25);
      input1[4 * 16 + i] = Eigen::half(-35.125);
      input1[5 * 16 + i] = Eigen::half(-70);
    }
  }

  // read data from file for input2
  constexpr uint64_t input2_size = 2;
  Eigen::half input2[input2_size] = {Eigen::half(-2), Eigen::half(-1)};

  constexpr uint64_t output_size = 1 * 1 * 2 * 3 * 16;
  Eigen::half output[output_size] = {half_zero};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};
  
  vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
  vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
  CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NC1HWC0, 1, use_default_theta, default_theta);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  Eigen::half output_exp[output_size] = {half_zero};
  for (uint32_t i = 0; i < 1; i ++) {
    if (i < 1) {
      output_exp[0 * 16 + i] = Eigen::half(0);
      output_exp[1 * 16 + i] = Eigen::half(-34.0);
      output_exp[2 * 16 + i] = Eigen::half(0);
      output_exp[3 * 16 + i] = Eigen::half(0);
      output_exp[4 * 16 + i] = Eigen::half(-12.302080);
      output_exp[5 * 16 + i] = Eigen::half(-15.499998);
    }
  }

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}
