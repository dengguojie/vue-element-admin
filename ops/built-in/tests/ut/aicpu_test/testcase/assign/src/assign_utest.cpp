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

class TEST_ASSIGN_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto output = node_def.MutableOutputs(0);
  int64_t output_num = output->NumElements();
  for (int i = 0; i < output_num; i++) {
    expect_out[i] = 1;
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Assign", "Assign")               \
      .Input({"ref", data_types[0], shapes[0], datas[0]})          \
      .Input({"value", data_types[1], shapes[1], datas[1]})        \
      .Output({"ref", data_types[2], shapes[2], datas[2]});

#define ASSIGN_CASE_WITH_SHAPE(case_name, base_type, aicpu_type,                  \
                               shapes, data_num)                                  \
  TEST_F(TEST_ASSIGN_UT, TestAssign_##case_name) {                                \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};           \
    base_type input1[data_num[0]] = {(base_type)1};                               \
    base_type input2[data_num[1]] = {(base_type)1};                               \
    base_type output[data_num[2]] = {(base_type)0};                               \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};      \
    CREATE_NODEDEF(shapes, data_types, datas);                                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                 \
    base_type expect_out[data_num[2]] = {(base_type)0};                           \
    CalcExpectFunc(*node_def.get(), expect_out);                                  \
    CompareResult<base_type>(output, expect_out, data_num[2]);                    \
  }

vector<vector<int64_t>> shapes_assign = {{2, 1, 1}, {2, 1, 1}, {2, 1, 1}};
vector<int64_t> data_num_assign = {2, 2, 2};
ASSIGN_CASE_WITH_SHAPE(assign, int64_t, DT_INT64, shapes_assign, data_num_assign)