#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "iostream"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_test_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#undef private
#undef protected

using namespace std;
using namespace aicpu;


class TEST_EXPONENTIAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, lambda, seed)        \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  if (seed == -1) {                                                    \
    NodeDefBuilder(node_def.get(), "Exponential", "Exponential")       \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("lambda", lambda);                                         \
  } else {                                                             \
    NodeDefBuilder(node_def.get(), "Exponential", "Exponential")       \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("lambda", lambda)                                          \
      .Attr("seed", seed);                                             \
  }

#define EXPONENTIAL_CASE_WITH_TYPE(case_name, input_type, input_aicpu_type,             \
                                   output_type, ouput_aicpu_type, lambda, seed)         \
  TEST_F(TEST_EXPONENTIAL_UT, TestExponential_##case_name) {                            \
    vector<DataType> data_types = {input_aicpu_type, ouput_aicpu_type};                 \
    input_type input[4] = {(input_type)2, (input_type)2, (input_type)2, (input_type)2}; \
    output_type output[4] = {(output_type)0};                                           \
    vector<void *> datas = {(void *)input, (void *)output};                             \
    vector<vector<int64_t>> shapes = {{2,2}, {2,2}};                                    \
    CREATE_NODEDEF(shapes, data_types, datas, lambda, seed);                            \
    if (input_aicpu_type != DT_FLOAT && input_aicpu_type != DT_DOUBLE &&                \
      input_aicpu_type != DT_FLOAT16) {                                                 \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                          \
    } else if (input_aicpu_type != ouput_aicpu_type || lambda <= 0) {                   \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                          \
    } else if (input_aicpu_type == ouput_aicpu_type) {                                  \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                     \
    }                                                                                   \
  }                                                                                     \

EXPONENTIAL_CASE_WITH_TYPE(float16_with_seed, Eigen::half, DT_FLOAT16, Eigen::half, DT_FLOAT16, 3.0, 10)

EXPONENTIAL_CASE_WITH_TYPE(float_with_seed, float, DT_FLOAT, float, DT_FLOAT, 5.0, 10)

EXPONENTIAL_CASE_WITH_TYPE(double_with_no_seed, double, DT_DOUBLE, double, DT_DOUBLE, 2.0, -1)

EXPONENTIAL_CASE_WITH_TYPE(input_with_ouput_not_match, double, DT_DOUBLE, float, DT_FLOAT, 1.0, -1)

EXPONENTIAL_CASE_WITH_TYPE(data_type_not_support, int16_t, DT_INT16, int16_t, DT_INT16, 1.0, -1)

EXPONENTIAL_CASE_WITH_TYPE(lambda_invalid, double, DT_DOUBLE, double, DT_DOUBLE, -1.1, -1)
