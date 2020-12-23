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
#include <algorithm>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_TOPK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "TopK", "TopK")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Input({"k", data_types[1], shapes[1], datas[1]})                        \
      .Output({"values", data_types[2], shapes[2], datas[2]})                  \
      .Output({"indices", data_types[3], shapes[3], datas[3]})                 \
      .Attr("sorted", true);

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_TOPK_UT, TestTopK_##aicpu_type) {                                \
    vector<DataType> data_types = {aicpu_type, DT_INT32, aicpu_type, DT_INT32};\
    vector<vector<int64_t>> shapes = {{24}, {}, {7}, {7}};                     \
    base_type input[24];                                                       \
    SetRandomValue<base_type>(input, 24);                                      \
    int32_t k = 7;                                                             \
    base_type output_value[7] = {(base_type)0};                                \
    int32_t output_index[7] = {0};                                             \
    vector<void *> datas = {(void *)input, (void *)&k, (void *)output_value,   \
                            (void *)output_index};                             \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
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
