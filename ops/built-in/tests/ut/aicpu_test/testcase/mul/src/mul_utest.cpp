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

class TEST_MUL_UT : public testing::Test {};

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
  NodeDefBuilder(node_def.get(), "Mul", "Mul")                     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]});

#define MUL_CASE_WITH_SHAPE(case_name, base_type, aicpu_type,                  \
                            shapes, data_num)                                  \
  TEST_F(TEST_MUL_UT, TestMul_##case_name) {                                   \
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

#define MUL_CASE_WITH_TYPE_DISMATCH(case_name, base_type1, base_type2,             \
                                    aicpu_type1, aicpu_type2, shapes, data_num)    \
  TEST_F(TEST_MUL_UT, TestMul_##case_name) {                                       \
    vector<DataType> data_types = {aicpu_type1, aicpu_type2, aicpu_type1};         \
    base_type1 input1[data_num[0]] = {(base_type1)1};                              \
    base_type2 input2[data_num[1]] = {(base_type2)1};                              \
    base_type1 output[data_num[2]] = {(base_type1)0};                              \
    vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};       \
    CREATE_NODEDEF(shapes, data_types, datas);                                     \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                       \
  }

vector<vector<int64_t>> shapes_mul1 = {{}, {}, {}};
vector<int64_t> data_num_mul1 = {1, 1, 1};
MUL_CASE_WITH_SHAPE(int8_scalar_mul_scalar, int8_t, DT_INT8, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_SHAPE(uint8_scalar_mul_scalar, uint8_t, DT_UINT8, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_SHAPE(uint16_scalar_mul_scalar, uint16_t, DT_UINT16, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_SHAPE(uint32_scalar_mul_scalar, uint32_t, DT_UINT32, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_SHAPE(uint64_scalar_mul_scalar, uint64_t, DT_UINT64, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_SHAPE(float_scalar_mul_scalar, float, DT_FLOAT, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_SHAPE(double_scalar_mul_scalar, double, DT_DOUBLE, shapes_mul1, data_num_mul1)

vector<vector<int64_t>> shapes_mul2 = {{}, {2}, {2}};
vector<int64_t> data_num_mul2 = {1, 2, 2};
MUL_CASE_WITH_SHAPE(int16_scalar_mul_vector, int16_t, DT_INT16, shapes_mul2, data_num_mul2)

vector<vector<int64_t>> shapes_mul3 = {{2}, {}, {2}};
vector<int64_t> data_num_mul3 = {2, 1, 2};
MUL_CASE_WITH_SHAPE(int32_vector_mul_scalar, int32_t, DT_INT32, shapes_mul3, data_num_mul3)

vector<vector<int64_t>> shapes_mul4 = {{2, 1, 1}, {2, 1, 1}, {2, 1, 1}};
vector<int64_t> data_num_mul4 = {2, 2, 2};
MUL_CASE_WITH_SHAPE(int64_vector_mul_vector_match, int64_t, DT_INT64, shapes_mul4, data_num_mul4)

vector<vector<int64_t>> shapes_mul5 = {{1}, {2, 1, 1, 1}, {2, 1, 1, 1}};
vector<int64_t> data_num_mul5 = {1, 2, 2};
MUL_CASE_WITH_SHAPE(int8_vector_mul_vector_broadcast_0, int8_t, DT_INT8, shapes_mul5, data_num_mul5)

vector<vector<int64_t>> shapes_mul6 = {{2, 1, 1, 1}, {1}, {2, 1, 1, 1}};
vector<int64_t> data_num_mul6 = {2, 1, 2};
MUL_CASE_WITH_SHAPE(int16_mul_vector_broadcast_1, int16_t, DT_INT16, shapes_mul6, data_num_mul6)

vector<vector<int64_t>> shapes_mul_dim7 = {{2, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1}};
vector<int64_t> data_num_mul_dim7 = {2, 1, 2};
MUL_CASE_WITH_SHAPE(int16_mul_vector_dim_7, int16_t, DT_INT16, shapes_mul_dim7, data_num_mul_dim7)

vector<vector<int64_t>> shapes_mul_dim8 = {{2, 1, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1, 1}};
vector<int64_t> data_num_mul_dim8 = {2, 1, 2};
MUL_CASE_WITH_SHAPE(int16_mul_vector_dim_8, int16_t, DT_INT16, shapes_mul_dim8, data_num_mul_dim8)

vector<vector<int64_t>> shapes_mul7 = {{1, 2}, {2, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 2}};
vector<int64_t> data_num_mul7 = {2, 2, 4};
MUL_CASE_WITH_SHAPE(int32_vector_mul_vector_broatcast_both, int32_t, DT_INT32, shapes_mul7, data_num_mul7)


MUL_CASE_WITH_TYPE_DISMATCH(bool_unsupport, bool, bool, DT_BOOL, DT_BOOL, shapes_mul1, data_num_mul1)

MUL_CASE_WITH_TYPE_DISMATCH(data_type_dismatch, float, double, DT_FLOAT, DT_DOUBLE, shapes_mul1, data_num_mul1)

vector<vector<int64_t>> shapes_mul8 = {{1}, {1}, {1, 1}};
vector<int64_t> data_num_mul8 = {1, 1, 1};
MUL_CASE_WITH_TYPE_DISMATCH(bcast_max_size_dismatch, float, float, DT_FLOAT, DT_FLOAT, shapes_mul8, data_num_mul8)

vector<vector<int64_t>> shapes_mul9 = {{1, 2}, {1, 3}, {1, 1}};
vector<int64_t> data_num_mul9 = {1, 1, 1};
MUL_CASE_WITH_TYPE_DISMATCH(bcast_shape_dismatch, float, float, DT_FLOAT, DT_FLOAT, shapes_mul9, data_num_mul9)

vector<vector<int64_t>> shapes_mul10 = {{1, 2}, {1, 3}, {1, 3}};
vector<int64_t> data_num_mul10 = {1, 1, 1};
MUL_CASE_WITH_TYPE_DISMATCH(bcast_not_support, float, float, DT_FLOAT, DT_FLOAT, shapes_mul10, data_num_mul10)