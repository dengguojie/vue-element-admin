#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include <cmath>

#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_DIAG_UT : public testing::Test {};

#define CREATE_DIAG_NODEDEF(shapes, data_types, datas)             \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Diag", "Diag")                   \
      .Input({"input", data_types[0], shapes[0], datas[0]})        \
      .Output({"output", data_types[1], shapes[1], datas[1]});

#define ADD_DIAG_CASE(base_type, aicpu_type)                             \
  TEST_F(TEST_DIAG_UT, TestDiag_##aicpu_type) {                          \
    vector<DataType> data_types = {aicpu_type, aicpu_type};              \
    vector<vector<int64_t>> shapes = {{2}, {4}};                         \
    base_type input[2] = {base_type(1), (base_type)2};                   \
    base_type output[4] = {(base_type)0};                                \
    vector<void *> datas = {(void *)input, (void *)output};              \
    CREATE_DIAG_NODEDEF(shapes, data_types, datas);                      \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    base_type expect_out[4] = {(base_type)1, (base_type)0, (base_type)0, \
                               (base_type)2};                            \
    EXPECT_EQ(CompareResult<base_type>(output, expect_out, 4), true);    \
  }

ADD_DIAG_CASE(Eigen::half, DT_FLOAT16)
ADD_DIAG_CASE(float, DT_FLOAT)
ADD_DIAG_CASE(double, DT_DOUBLE)
ADD_DIAG_CASE(int32_t, DT_INT32)
ADD_DIAG_CASE(int64_t, DT_INT64)
ADD_DIAG_CASE(std::complex<float>, DT_COMPLEX64)
ADD_DIAG_CASE(std::complex<double>, DT_COMPLEX128)

#define CREATE_DIAG_PART_NODEDEF(shapes, data_types, datas)        \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "DiagPart", "DiagPart")           \
      .Input({"input", data_types[0], shapes[0], datas[0]})        \
      .Output({"output", data_types[1], shapes[1], datas[1]});

#define ADD_DIAG_PART_CASE(base_type, aicpu_type)                     \
  TEST_F(TEST_DIAG_UT, TestDiagPart_##aicpu_type) {                   \
    vector<DataType> data_types = {aicpu_type, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{2, 2}, {2}};                   \
    base_type input[4] = {(base_type)1, (base_type)0, (base_type)0,   \
                          (base_type)2};                              \
    base_type output[2] = {(base_type)0};                             \
    vector<void *> datas = {(void *)input, (void *)output};           \
    CREATE_DIAG_PART_NODEDEF(shapes, data_types, datas);              \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                     \
    base_type expect_out[2] = {(base_type)1, (base_type)2};           \
    EXPECT_EQ(CompareResult<base_type>(output, expect_out, 2), true); \
  }

ADD_DIAG_PART_CASE(Eigen::half, DT_FLOAT16)
ADD_DIAG_PART_CASE(float, DT_FLOAT)
ADD_DIAG_PART_CASE(double, DT_DOUBLE)
ADD_DIAG_PART_CASE(int32_t, DT_INT32)
ADD_DIAG_PART_CASE(int64_t, DT_INT64)
ADD_DIAG_PART_CASE(std::complex<float>, DT_COMPLEX64)
ADD_DIAG_PART_CASE(std::complex<double>, DT_COMPLEX128)
