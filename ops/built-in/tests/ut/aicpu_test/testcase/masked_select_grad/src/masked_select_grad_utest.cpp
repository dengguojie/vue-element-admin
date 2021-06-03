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

class TEST_MASKED_SELECT_GRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                           \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();          \
  NodeDefBuilder(node_def.get(), "MaskedSelectGrad", "MaskedSelectGrad")    \
      .Input({"x", data_types[0], shapes[0], datas[0]})                     \
      .Input({"mask", data_types[1], shapes[1], datas[1]})                  \
      .Input({"grad", data_types[2], shapes[2], datas[2]})                  \
      .Output({"dx", data_types[3], shapes[3], datas[3]})

#define ADD_CASE(base_type, aicpu_type)                                                          \
  TEST_F(TEST_MASKED_SELECT_GRAD_UT, TestMaskedSelectGrad_##aicpu_type) {                        \
    vector<DataType> data_types = {aicpu_type, DT_BOOL, aicpu_type, aicpu_type};                 \
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2}, {2, 2}};                              \
    base_type x[4] = {base_type(1), base_type(2), base_type(3), base_type(4)};                   \
    bool mask[4] = {0, 1, 0, 1};                                                                 \
    base_type grad[2] = {base_type(7), base_type(8)};                                            \
    base_type dx[4] = {base_type(0), base_type(0), base_type(0), base_type(0)};                  \
    vector<void *> datas = {(void *)x, (void *)mask, (void *)grad, (void *)dx};                  \
    CREATE_NODEDEF(shapes, data_types, datas);                                                   \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                \
    EXPECT_EQ(dx[1], base_type(7));                                                              \
    EXPECT_EQ(dx[3], base_type(8));                                                              \
  }

TEST_F(TEST_MASKED_SELECT_GRAD_UT, UNDEFINED_DTYPE_EXCEPTION) {
    vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
    vector<DataType> data_types = {DT_UNDEFINED, DT_UNDEFINED, DT_UNDEFINED, DT_UNDEFINED};
    vector<int32_t> input1 = {1};
    vector<int32_t> input2 = {1};
    vector<int32_t> input3 = {1};
    vector<int32_t> output = {0};
    vector<void *> datas = { (void *)input1.data(),
                             (void *)input2.data(),
                             (void *)input3.data(),
                             (void *)output.data() };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MASKED_SELECT_GRAD_UT, INPUT_DTYPE_DISMATCH) {
    vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
    float input1[1] = {1.0F};
    bool input2[1] = {1};
    double input3[1] = {2.0};
    double output[1] = {0.0};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)input3,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MASKED_SELECT_GRAD_UT, BROADCAST_NOT_SUPPORT) {
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {}, {1, 3}};
    vector<DataType> data_types = {DT_FLOAT, DT_BOOL, DT_FLOAT, DT_FLOAT};
    float input1[2] = {1.0F};
    bool input2[3] = {1};
    float input3[2] = {1.0F};
    float output[3] = {0.0F};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)input3,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

ADD_CASE(int8_t, DT_INT8)

ADD_CASE(int16_t, DT_INT16)

ADD_CASE(int32_t, DT_INT32)

ADD_CASE(int64_t, DT_INT64)

ADD_CASE(uint8_t, DT_UINT8)

ADD_CASE(uint16_t, DT_UINT16)

ADD_CASE(uint32_t, DT_UINT32)

ADD_CASE(uint64_t, DT_UINT64)

ADD_CASE(bool, DT_BOOL)
