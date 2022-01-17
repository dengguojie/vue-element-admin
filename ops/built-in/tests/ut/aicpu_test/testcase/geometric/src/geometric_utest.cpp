#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_GEOMETRIC_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, p, seed)             \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  if (seed == -1) {                                                    \
    NodeDefBuilder(node_def.get(), "Geometric", "Geometric")           \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("p", p);                                                   \
  } else {                                                             \
    NodeDefBuilder(node_def.get(), "Geometric", "Geometric")           \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("p", p)                                                    \
      .Attr("seed", seed);                                             \
  }

#define GEOMETRIC_CASE_WITH_TYPE(case_name, base_type, aicpu_data_type,                  \
                                      aicpu_dtype, seed)                                 \
  TEST_F(TEST_GEOMETRIC_UT, TestGeometric_##case_name) {                                 \
    vector<DataType> data_types = {aicpu_data_type, aicpu_dtype};                        \
    int64_t input[2] = {2, 2};                                                           \
    base_type output[4] = {(base_type)0};                                                \
    vector<void *> datas = {(void *)input, (void *)output};                              \
    vector<vector<int64_t>> shapes = {{2, 2}, {input[0], input[1]}};                     \
    CREATE_NODEDEF(shapes, data_types, datas, 0.2, seed);                                \
    if (aicpu_dtype != DT_FLOAT && aicpu_dtype != DT_FLOAT16 ) {                         \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                           \
    } else if (aicpu_dtype != aicpu_dtype) {                                             \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                           \
    } else if (aicpu_data_type == aicpu_dtype) {                                         \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                      \
    }                                                                                    \
  }

GEOMETRIC_CASE_WITH_TYPE(float16_case, Eigen::half, DT_FLOAT16, DT_FLOAT16, 10)

GEOMETRIC_CASE_WITH_TYPE(float, float, DT_FLOAT, DT_FLOAT, 10)

GEOMETRIC_CASE_WITH_TYPE(with_no_seed, float, DT_FLOAT, DT_FLOAT, -1)

// exception instance
TEST_F(TEST_GEOMETRIC_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3, 3}, {3, 4, 2}};
  int32_t input[27] = {(int32_t)1};
  int32_t output[24] = {(int32_t)1};
  vector<void *> datas = {(void *)input,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0.2, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GEOMETRIC_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  bool input[6] = {(bool)1};
  bool output[6] = {(bool)0};
  vector<void *> datas = {(void *)input, 
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0.2, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID); 
}

TEST_F(TEST_GEOMETRIC_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool output[22] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0.2, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GEOMETRIC_UT, INPUT_P_UNSUPPORT) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool input[22] = {(float)1};
  bool output[22] = {(float)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 1.2, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
