#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif

#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_CLIP_BY_VALUE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "ClipByValue", "ClipByValue")                       \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Input({"clip_value_min", data_types[1], shapes[1], datas[1]})            \
      .Input({"clip_value_max", data_types[2], shapes[2], datas[2]})            \
      .Output({"y", data_types[3], shapes[3], datas[3]}) ;

TEST_F(TEST_CLIP_BY_VALUE_UT, TestCLIP_BY_VALUE_INVALID_FLOAT16) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1,6}, {1,6}, {1,6}, {1,6}};
  Eigen::half input0[6] = {(Eigen::half)11.1, (Eigen::half)22.2, (Eigen::half)33.3, (Eigen::half)44.4, (Eigen::half)55.5, (Eigen::half)66.6};
  Eigen::half input1[6] = {(Eigen::half)11.1, (Eigen::half)22.2, (Eigen::half)33.3, (Eigen::half)44.4, (Eigen::half)55.5, (Eigen::half)66.6};
  Eigen::half input2[6] = {(Eigen::half)11.1, (Eigen::half)22.2, (Eigen::half)33.3, (Eigen::half)44.4, (Eigen::half)55.5, (Eigen::half)66.6};
  Eigen::half output0[6] = {(Eigen::half)0};
  Eigen::half expect_output0[6] = {(Eigen::half)11.1, (Eigen::half)22.2, (Eigen::half)33.3, (Eigen::half)44.4, (Eigen::half)55.5, (Eigen::half)66.6};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CLIP_BY_VALUE_UT, TestCLIP_BY_VALUE_SUCCESS_FLOAT_1) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1,6}, {1,6}, {1,6}, {1,6}};
  float input0[6] = {(float)11.1, (float)22.2, (float)30.3, (float)54.4, (float)55.5, (float)66.6};
  float input1[6] = {(float)11.1, (float)12.2, (float)33.3, (float)40.4, (float)55.5, (float)66.6};
  float input2[6] = {(float)11.1, (float)32.2, (float)53.3, (float)44.4, (float)55.5, (float)66.6};
  float expect_output0[6] = {(float)11.1, (float)22.2, (float)33.3, (float)44.4, (float)55.5, (float)66.6};
  float output0[6] = {(float)0};  
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  EXPECT_EQ(CompareResult<float>(output0, expect_output0, 6), true);
}

TEST_F(TEST_CLIP_BY_VALUE_UT, TestCLIP_BY_VALUE_SUCCESS_FLOAT_2) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1,6}, {}, {}, {1,6}};
  float input0[6] = {(float)11.1, (float)22.2, (float)33.3, (float)44.4, (float)55.5, (float)66.6};
  float input1 = (float)35.1;
  float input2 = (float)56.1;
  float expect_output0[6] = {(float)35.1, (float)35.1, (float)35.1, (float)44.4, (float)55.5, (float)56.1};
  float output0[6] = {(float)0};  
  vector<void *> datas = {(void *)input0, (void *)&input1, (void *)&input2, (void *)output0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  EXPECT_EQ(CompareResult<float>(output0, expect_output0, 6), true);
}

TEST_F(TEST_CLIP_BY_VALUE_UT, TestCLIP_BY_VALUE_SUCCESS_DOUBLE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1,6}, {1,6}, {1,6}, {1,6}};
  double input0[6] = {(double)11.1, (double)22.2, (double)30.3, (double)54.4, (double)55.5, (double)66.6};
  double input1[6] = {(double)11.1, (double)12.2, (double)33.3, (double)40.4, (double)55.5, (double)66.6};
  double input2[6] = {(double)11.1, (double)32.2, (double)53.3, (double)44.4, (double)55.5, (double)66.6};
  double expect_output0[6] = {(double)11.1, (double)22.2, (double)33.3, (double)44.4, (double)55.5, (double)66.6};
  double output0[6] = {(double)0};  
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output0};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  EXPECT_EQ(CompareResult<double>(output0, expect_output0, 6), true);
}




