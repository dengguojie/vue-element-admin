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
#include <math.h>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_REALDIV_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto output = node_def.MutableOutputs(0);
  int64_t output_num = output->NumElements();
  for (int i = 0; i < output_num; i++) {
    expect_out[i] = 1;
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                                 \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                \
  NodeDefBuilder(node_def.get(), "RealDiv", "RealDiv")                            \
      .Input({"x1", data_types[0], shapes[0], datas[0]})                          \
      .Input({"x2", data_types[1], shapes[1], datas[1]})                          \
      .Output({"y", data_types[2], shapes[2], datas[2]})

#define ADD_CASE(base_type, aicpu_type)                                           \
  TEST_F(TEST_REALDIV_UT, TestRealDiv_##aicpu_type) {                             \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{24}, {24}, {24}};                          \
    base_type input_x1[24];                                                       \
    SetRandomValue<base_type>(input_x1, 24);                                      \
    base_type input_x2[24];                                                       \
    SetRandomValue<base_type>(input_x2, 24, 1);                                   \
    base_type output[24] = {(base_type)0};                                        \
    vector<void *> datas = {(void *)input_x1, (void *)input_x2, (void *)output};  \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                 \
  }

#define REALDIV_CASE_WITH_SHAPE(case_name, base_type, aicpu_type,                  \
                                shapes, data_num)                                  \
  TEST_F(TEST_REALDIV_UT, TestRealdiv_##case_name) {                               \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};            \
    base_type input1[data_num[0]] = {(base_type)1};                                \
    base_type input2[data_num[1]] = {(base_type)1};                                \
    base_type output[data_num[2]] = {(base_type)0};                                \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};       \
    CREATE_NODEDEF(shapes, data_types, datas);                                     \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                  \
    base_type expect_out[data_num[2]] = {(base_type)0};                            \
    CalcExpectFunc(*node_def.get(), expect_out);                                   \
    CompareResult<base_type>(output, expect_out, data_num[2]);                     \
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

vector<vector<int64_t>> shapes_realdiv_dim7 = {{2, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1}};
vector<int64_t> data_num_realdiv_dim7 = {2, 1, 2};
REALDIV_CASE_WITH_SHAPE(int16_realdiv_vector_dim_7, int16_t, DT_INT16, shapes_realdiv_dim7, data_num_realdiv_dim7)

vector<vector<int64_t>> shapes_realdiv_dim8 = {{2, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1}};
vector<int64_t> data_num_realdiv_dim8 = {2, 1, 2};
REALDIV_CASE_WITH_SHAPE(int16_realdiv_vector_dim_8, int16_t, DT_INT16, shapes_realdiv_dim8, data_num_realdiv_dim8)