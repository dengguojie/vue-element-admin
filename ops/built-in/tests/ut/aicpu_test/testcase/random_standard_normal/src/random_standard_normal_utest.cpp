#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_test_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#undef private
#undef protected

using namespace std;
using namespace aicpu;


class TEST_RANDOM_STANDARD_NORMAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2)        \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  if (seed1 == -1) {                                                   \
    NodeDefBuilder(node_def.get(), "RandomStandardNormal", "RandomStandardNormal")   \
      .Input({"shape", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("dtype", data_types[2]);                                   \
  } else if (seed2 == -1) {                                            \
    NodeDefBuilder(node_def.get(), "RandomStandardNormal", "RandomStandardNormal")   \
      .Input({"shape", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("seed", seed1)                                             \
      .Attr("dtype", data_types[2]);                                   \
  } else {                                                             \
    NodeDefBuilder(node_def.get(), "RandomStandardNormal", "RandomStandardNormal")   \
      .Input({"shape", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("seed", seed1)                                             \
      .Attr("seed2", seed2)                                            \
      .Attr("dtype", data_types[2]);                                   \
  }


#define RANDOM_STANDARD_NORMAL_CASE_WITH_TYPE(case_name, base_type, aicpu_data_type,             \
                                      aicpu_dtype, seed1, seed2)                         \
  TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, TestRandomStandardNormal_##case_name) {                        \
    vector<DataType> data_types = {DT_INT64, aicpu_data_type, aicpu_dtype};              \
    int64_t input[2] = {2, 2};                                                           \
    base_type output[4] = {(base_type)0};                                                \
    vector<void *> datas = {(void *)input, (void *)output};                              \
    vector<vector<int64_t>> shapes = {{2}, {input[0], input[1]}};                        \
    CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2);                             \
    if (aicpu_dtype != DT_FLOAT && aicpu_dtype != DT_DOUBLE &&                           \
        aicpu_dtype != DT_FLOAT16) {                                                     \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                           \
    } else if (aicpu_dtype != aicpu_dtype) {                                             \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                           \
    } else if (aicpu_data_type == aicpu_dtype) {                                         \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                      \
    }                                                                                    \
  }

RANDOM_STANDARD_NORMAL_CASE_WITH_TYPE(float16_with_seed, Eigen::half, DT_FLOAT16, DT_FLOAT16, 10, -1)

RANDOM_STANDARD_NORMAL_CASE_WITH_TYPE(float_with_seed2, float, DT_FLOAT, DT_FLOAT, 10, 5)

RANDOM_STANDARD_NORMAL_CASE_WITH_TYPE(double_with_no_seed, double, DT_DOUBLE, DT_DOUBLE, -1, -1)

RANDOM_STANDARD_NORMAL_CASE_WITH_TYPE(data_type_not_match, double, DT_DOUBLE, DT_INT64, -1, -1)

RANDOM_STANDARD_NORMAL_CASE_WITH_TYPE(data_type_not_support, int64_t, DT_INT64, DT_INT64, -1, -1)

