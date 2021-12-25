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

class TEST_POISSON_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, seed)                \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();     \
  if (seed == -1) {                                                    \
    NodeDefBuilder(node_def.get(), "Poisson", "Poisson")               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]});              \                                                 
  } else {                                                             \
    NodeDefBuilder(node_def.get(), "Poisson", "Poisson")               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                \
      .Output({"y", data_types[1], shapes[1], datas[1]})               \
      .Attr("seed", seed);                                             \
  }

#define POISSON_CASE_WITH_TYPE(case_name, base_type, aicpu_data_type,                    \
                                      aicpu_dtype, seed)                                 \
  TEST_F(TEST_POISSON_UT, TestPOISSON_##case_name) {                                     \
    vector<DataType> data_types = {aicpu_data_type, aicpu_dtype};                        \
    base_type input[2] = {(base_type)2};                                                 \
    base_type output[4] = {(base_type)0};                                                \
    vector<void *> datas = {(void *)input, (void *)output};                              \
    vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};                                   \
    CREATE_NODEDEF(shapes, data_types, datas, seed);                                     \
    if (aicpu_dtype != DT_FLOAT && aicpu_dtype != DT_FLOAT16 ) {                         \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                           \
    } else if (aicpu_dtype != aicpu_dtype) {                                             \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                           \
    } else if (aicpu_data_type == aicpu_dtype) {                                         \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                      \
    }                                                                                    \
  }

POISSON_CASE_WITH_TYPE(float16_case, Eigen::half, DT_FLOAT16, DT_FLOAT16, 10)

POISSON_CASE_WITH_TYPE(float, float, DT_FLOAT, DT_FLOAT, 10)

POISSON_CASE_WITH_TYPE(with_no_seed, float, DT_FLOAT, DT_FLOAT, 10)
  
TEST_F(TEST_POISSON_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
  float x[4] = {(float)2};
  float output[4] = {(float)0};
  vector<void *> datas = {
                          (void *)x,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_POISSON_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
  Eigen::half x[4] = {(Eigen::half)2};
  Eigen::half output[4] = {(Eigen::half)1};
  vector<void *> datas = {
                          (void *)x,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

// exception instance
TEST_F(TEST_POISSON_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
  float output[4] = {(float)0};
  vector<void *> datas = {
                          (void *)nullptr,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_POISSON_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}};
  bool x[4] = {(bool)1};
  bool output[4] = {(bool)0};
  vector<void *> datas = {(void *)x, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 10);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}