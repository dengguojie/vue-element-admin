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

class TEST_FILL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();    \
  NodeDefBuilder(node_def.get(), "Fill", "Fill")                      \
      .Input({"dims", data_types[0], shapes[0], datas[0]})            \
      .Input({"value", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]});

#define FILL_INVALID_DIMS_CASE(case_name, dims_shape, dims_dtype, dims_data)  \
  TEST_F(TEST_FILL_UT, TestFill_##case_name) {                                \
    std::vector<std::vector<int64_t>> shapes = {dims_shape, {}, {1, 1}};      \
    std::vector<DataType> data_types = { dims_dtype, DT_INT32, DT_INT32 };    \
    std::vector<int32_t> value = {1};                                         \
    std::vector<int32_t> output = {0};                                        \
    std::vector<void *> datas = { (void *)dims_data.data(),                   \
                                  (void *)value.data(),                       \
                                  (void *)output.data() };                    \
    CREATE_NODEDEF(shapes, data_types, datas)                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)                   \
  }

#define FILL_INVALID_OUTPUT_CASE(case_name, output_shape, output_dtype, output_data)  \
  TEST_F(TEST_FILL_UT, TestFill_##case_name) {                                        \
    std::vector<std::vector<int64_t>> shapes = {{2}, {}, output_shape};               \
    std::vector<DataType> data_types = { DT_INT64, DT_INT32, output_dtype };          \
    std::vector<int64_t> dims = {1, 1};                                               \
    std::vector<int32_t> value = {1};                                                 \
    std::vector<void *> datas = { (void *)dims.data(),                                \
                                  (void *)value.data(),                               \
                                  (void *)output_data.data() };                       \
    CREATE_NODEDEF(shapes, data_types, datas)                                         \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)                           \
  }

#define FILL_VALID_OUTPUT_CASE(case_name, dtype, T)                                   \
  TEST_F(TEST_FILL_UT, TestFill_##case_name) {                                        \
    std::vector<std::vector<int64_t>> shapes = {{2}, {}, {1, 1}};                     \
    std::vector<DataType> data_types = { DT_INT64, dtype, dtype };                    \
    int64_t dims[2] = {1, 1};                                                         \
    T value[1] = {(T)1};                                                              \
    T output[1] = {(T)0};                                                             \
    std::vector<void *> datas = { (void *)dims,                                       \
                                  (void *)value,                                      \
                                  (void *)output };                                   \
    CREATE_NODEDEF(shapes, data_types, datas)                                         \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK)                                      \
  }

std::vector<int64_t> dims_shape1 = {2};
std::vector<uint32_t> dims1 = {1, 1};
FILL_INVALID_DIMS_CASE(unsupport_dims_dtype, dims_shape1, DT_UINT32, dims1)

std::vector<int64_t> dims_shape2 = {};
std::vector<int32_t> dims2 = {};
FILL_INVALID_DIMS_CASE(empty_dims, dims_shape2, DT_INT32, dims2)

std::vector<int64_t> dims_shape3 = {2};
std::vector<int32_t> dims3 = {1, -1};
FILL_INVALID_DIMS_CASE(negative_dims, dims_shape3, DT_INT32, dims3)

std::vector<int64_t> dims_shape4 = {2};
std::vector<int32_t> dims4 = {1, 0};
FILL_INVALID_DIMS_CASE(zero_dims, dims_shape4, DT_INT32, dims4)

TEST_F(TEST_FILL_UT, not_scalar_value) {
  std::vector<std::vector<int64_t>> shapes = {{2}, {1}, {1, 1}};
  std::vector<DataType> data_types = { DT_INT64, DT_INT64, DT_INT64 };
  std::vector<int64_t> dims = {1, 1};
  std::vector<int64_t> value = {1};
  std::vector<int64_t> output = {0};
  std::vector<void *> datas = { (void *)dims.data(),
                                (void *)value.data(),
                                (void *)output.data() };
  CREATE_NODEDEF(shapes, data_types, datas)
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

std::vector<int64_t> output_shape1 = {2};
std::vector<int32_t> output1 = {1, 2};
FILL_INVALID_OUTPUT_CASE(shape_mismatch, output_shape1, DT_INT32, output1)

std::vector<int64_t> output_shape2 = {2};
std::vector<int64_t> output2 = {1, 1};
FILL_INVALID_OUTPUT_CASE(dtype_mismatch, output_shape2, DT_INT64, output2)

std::vector<int64_t> output_shape3 = {2};
std::vector<int64_t> output3 = {1, 1};
FILL_INVALID_OUTPUT_CASE(unsupport_output_dtype, output_shape3, DT_COMPLEX64, output3)

FILL_VALID_OUTPUT_CASE(int8_dtype, DT_INT8, int8_t)
FILL_VALID_OUTPUT_CASE(uint8_dtype, DT_UINT8, uint8_t)
FILL_VALID_OUTPUT_CASE(int16_dtype, DT_INT16, int16_t)
FILL_VALID_OUTPUT_CASE(uint16_dtype, DT_UINT16, uint16_t)
FILL_VALID_OUTPUT_CASE(int32_dtype, DT_INT32, int32_t)
FILL_VALID_OUTPUT_CASE(uint32_dtype, DT_UINT32, uint32_t)
FILL_VALID_OUTPUT_CASE(int64_dtype, DT_INT64, int64_t)
FILL_VALID_OUTPUT_CASE(uint64_dtype, DT_UINT64, uint64_t)
FILL_VALID_OUTPUT_CASE(bool_dtype, DT_BOOL, bool)
FILL_VALID_OUTPUT_CASE(float16_dtype, DT_FLOAT16, Eigen::half)
FILL_VALID_OUTPUT_CASE(float_dtype, DT_FLOAT, float)
FILL_VALID_OUTPUT_CASE(double_dtype, DT_DOUBLE, double)
