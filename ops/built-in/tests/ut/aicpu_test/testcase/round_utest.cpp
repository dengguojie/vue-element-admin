#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include <cmath>
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_ROUND_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  int64_t input0_num = input0->NumElements();
  for (int64_t i = 0; i < input0_num; ++i) {
    expect_out[i] = std::round(input0_data[i]);
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Round", "Round")                 \
      .Input({"input", data_types[0], shapes[0], datas[0]})        \
      .Output({"output", data_types[1], shapes[1], datas[1]});

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_ROUND_UT, TestRound_##aicpu_type) {                              \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                    \
    vector<vector<int64_t>> shapes = {{4}, {4}};                               \
    base_type input[4];                                                        \
    SetRandomValue<base_type>(input, 4);                                       \
    base_type output[4] = {(base_type)0};                                      \
    vector<void *> datas = {(void *)input, (void *)output};                    \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[4] = {(base_type)0};                                  \
    CalcExpectFunc(*node_def.get(), expect_out);                               \
    CompareResult<base_type>(output, expect_out, 44);                          \
  }

ADD_CASE(float, DT_FLOAT)
