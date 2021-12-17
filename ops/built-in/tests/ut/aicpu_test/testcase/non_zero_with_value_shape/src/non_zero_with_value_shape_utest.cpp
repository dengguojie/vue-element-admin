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
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_NON_ZERO_WITH_VALUE_SHAPE_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T &dims0, T &dims1) {
    auto shape0 = node_def.MutableOutputs(0)->GetTensorShape();
    auto shape1 = node_def.MutableOutputs(1)->GetTensorShape();
    dims0 = shape0->GetDimSizes();
    dims1 = shape1->GetDimSizes();
}

#define CREATE_NODEDEF(shapes, data_types, datas)                                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "NonZeroWithValueShape", "NonZeroWithValueShape") \
      .Input({"value", data_types[0], shapes[0], datas[0]})                        \
      .Input({"index", data_types[1], shapes[1], datas[1]})                        \
      .Input({"count", data_types[2], shapes[2], datas[2]})                        \
      .Output({"out_value", data_types[3], shapes[3], datas[3]})                   \
      .Output({"out_index", data_types[4], shapes[4], datas[4]})                 
 
TEST_F(TEST_NON_ZERO_WITH_VALUE_SHAPE_UT, DATA_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32,
                                 DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {4}, {1}, {}, {}};

  int32_t input0[2] = {1, 2};

  int32_t input1[4] = {1, 2, 3, 4};
  int32_t input2[1] = {2};
  int32_t output0[2] = {1, 2};
  int32_t output1[4] = {1, 2, 3, 4};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output0,
                          (void *)output1};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::vector<int64_t> dims0;
  std::vector<int64_t> dims1;
  CalcExpectFunc(*node_def.get(), dims0, dims1);
  std::vector<int64_t> output_exp0_shape_dims = {2};
  std::vector<int64_t> output_exp1_shape_dims = {2, 2};
  EXPECT_EQ(dims0, output_exp0_shape_dims);
  EXPECT_EQ(dims1, output_exp1_shape_dims);
}