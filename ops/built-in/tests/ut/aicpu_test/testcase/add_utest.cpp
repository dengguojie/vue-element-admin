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

class TEST_ADD_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef &node_def, T expect_out[]) {
  auto output = node_def.MutableOutputs(0);
  int64_t output_num = output->NumElements();
  for (int i = 0; i < output_num; i++) {
    expect_out[i] = 2;
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Add", "Add")                     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]});

#define ADD_CASE_WITH_SHAPE(case_name, base_type, aicpu_type,                  \
                            shapes, data_num)                                  \
  TEST_F(TEST_ADD_UT, TestAdd_##case_name) {                                   \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};        \
    base_type input1[data_num[0]] = {(base_type)1};                            \
    base_type input2[data_num[1]] = {(base_type)1};                            \
    base_type output[data_num[2]] = {(base_type)0};                            \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};   \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[data_num[2]] = {(base_type)0};                        \
    CalcExpectFunc(*node_def.get(), expect_out);                               \
    CompareResult<base_type>(output, expect_out, data_num[2]);                 \
  }

#define ADD_CASE_WITH_TYPE_DISMATCH(case_name, base_type1, base_type2,             \
                                    aicpu_type1, aicpu_type2, shapes, data_num)    \
  TEST_F(TEST_ADD_UT, TestAdd_##case_name) {                                       \
    vector<DataType> data_types = {aicpu_type1, aicpu_type2, aicpu_type1};         \
    base_type1 input1[data_num[0]] = {(base_type1)1};                              \
    base_type2 input2[data_num[1]] = {(base_type2)1};                              \
    base_type1 output[data_num[2]] = {(base_type1)0};                              \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};       \
    CREATE_NODEDEF(shapes, data_types, datas);                                     \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                       \
  }                                                                                \

vector<vector<int64_t>> shapes1 = {{}, {}, {}};
vector<int64_t> data_num1 = {1, 1, 1};
ADD_CASE_WITH_SHAPE(int8_scalar_add_scalar, int8_t, DT_INT8, shapes1, data_num1)

vector<vector<int64_t>> shapes2 = {{}, {2}, {2}};
vector<int64_t> data_num2 = {1, 2, 2};
ADD_CASE_WITH_SHAPE(int16_scalar_add_vector, int16_t, DT_INT16, shapes2, data_num2)

vector<vector<int64_t>> shapes3 = {{2}, {}, {2}};
vector<int64_t> data_num3 = {2, 1, 2};
ADD_CASE_WITH_SHAPE(int32_vector_add_scalar, int32_t, DT_INT32, shapes3, data_num3)

vector<vector<int64_t>> shapes4 = {{2, 1, 1}, {2, 1, 1}, {2, 1, 1}};
vector<int64_t> data_num4 = {2, 2, 2};
ADD_CASE_WITH_SHAPE(int64_vector_add_vector_match, int64_t, DT_INT64, shapes4, data_num4)

vector<vector<int64_t>> shapes5 = {{1}, {2, 1, 1, 1}, {2, 1, 1, 1}};
vector<int64_t> data_num5 = {1, 2, 2};
ADD_CASE_WITH_SHAPE(int8_vector_add_vector_broadcast_0, int8_t, DT_INT8, shapes5, data_num5)

vector<vector<int64_t>> shapes6 = {{2, 1, 1, 1}, {1}, {2, 1, 1, 1}};
vector<int64_t> data_num6 = {2, 1, 2};
ADD_CASE_WITH_SHAPE(int16_add_vector_broadcast_1, int16_t, DT_INT16, shapes6, data_num6)

vector<vector<int64_t>> shapes7 = {{1, 2}, {2, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 2}};
vector<int64_t> data_num7 = {2, 2, 4};
ADD_CASE_WITH_SHAPE(int32_vector_add_vector_broatcast_both, int32_t, DT_INT32, shapes7, data_num7)

ADD_CASE_WITH_TYPE_DISMATCH(bool_unsupport, bool, bool, DT_BOOL, DT_BOOL, shapes1, data_num1)

ADD_CASE_WITH_TYPE_DISMATCH(data_type_dismatch, float, double, DT_FLOAT, DT_DOUBLE, shapes1, data_num1)

vector<vector<int64_t>> shapes8 = {{1}, {1}, {1, 1}};
vector<int64_t> data_num8 = {1, 1, 1};
ADD_CASE_WITH_TYPE_DISMATCH(bcast_max_size_dismatch, float, float, DT_FLOAT, DT_FLOAT, shapes8, data_num8)

vector<vector<int64_t>> shapes9 = {{1, 2}, {1, 3}, {1, 1}};
vector<int64_t> data_num9 = {1, 1, 1};
ADD_CASE_WITH_TYPE_DISMATCH(bcast_shape_dismatch, float, float, DT_FLOAT, DT_FLOAT, shapes9, data_num9)

vector<vector<int64_t>> shapes10 = {{1, 2}, {1, 3}, {1, 3}};
vector<int64_t> data_num10 = {1, 1, 1};
ADD_CASE_WITH_TYPE_DISMATCH(bcast_not_support, float, float, DT_FLOAT, DT_FLOAT, shapes10, data_num10)