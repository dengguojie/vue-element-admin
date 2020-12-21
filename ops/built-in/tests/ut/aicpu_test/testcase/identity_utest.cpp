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

class TEST_IDENTITY_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();  \
  NodeDefBuilder(node_def.get(), "Identity", "Identity")            \
      .Input({"x", data_types[0], shapes[0], datas[0]})             \
      .Output({"y", data_types[1], shapes[1], datas[1]})

#define ADD_CASE(base_type, aicpu_type)                                     \
  TEST_F(TEST_IDENTITY_UT, TestIdentifyBroad_##aicpu_type) {                \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                 \
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};                    \
    vector<base_type> input(22);                                            \
    SetRandomValue<base_type>(input.data(), 22);                            \
    vector<base_type> output(22);                                           \
    vector<void *> datas = {(void *)input.data(), (void *)output.data()};   \
    CREATE_NODEDEF(shapes, data_types, datas);                              \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                           \
    vector<base_type> expect_out = input;                                   \
    EXPECT_EQ(output, expect_out);                                          \
  }

TEST_F(TEST_IDENTITY_UT, ExpSizeNotMatch) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}};
  vector<int32_t> input(16);
  vector<int32_t> output(12);
  vector<void *> datas = {(void *)input.data(), (void *)output.data()};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IDENTITY_UT, ExpInputNull) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  vector<int32_t> output(22);
  vector<void *> datas = {(void *)nullptr, (void *)output.data()};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_IDENTITY_UT, ExpOutputNull) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  vector<int32_t> input(22);
  vector<void *> datas = {(void *)input.data(), (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(uint16_t, DT_UINT16)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(uint64_t, DT_UINT64)
