#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include <cmath>

#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_ONES_LIKE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "OnesLike", "OnesLike")           \
      .Input({"input", data_types[0], shapes[0], datas[0]})        \
      .Output({"output", data_types[1], shapes[1], datas[1]});

#define ADD_CASE(base_type, aicpu_type)                                  \
  TEST_F(TEST_ONES_LIKE_UT, TestOnesLike_##aicpu_type) {                 \
    vector<DataType> data_types = {aicpu_type, aicpu_type};              \
    vector<vector<int64_t>> shapes = {{4}, {4}};                         \
    base_type input[4];                                                  \
    SetRandomValue<base_type>(input, 4);                                 \
    base_type output[4] = {(base_type)0};                                \
    vector<void *> datas = {(void *)input, (void *)output};              \
    CREATE_NODEDEF(shapes, data_types, datas);                           \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    base_type expect_out[4] = {(base_type)1, (base_type)1, (base_type)1, \
                               (base_type)1};                            \
    EXPECT_EQ(CompareResult<base_type>(output, expect_out, 4), true);    \
  }

ADD_CASE(float, DT_FLOAT)
ADD_CASE(double, DT_DOUBLE)
ADD_CASE(int8_t, DT_INT8)
ADD_CASE(int16_t, DT_INT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
ADD_CASE(uint8_t, DT_UINT8)
ADD_CASE(uint16_t, DT_UINT16)
