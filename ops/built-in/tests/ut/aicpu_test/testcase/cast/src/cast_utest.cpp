#include "gtest/gtest.h"
#include <math.h>
#include <stdint.h>
#include <Eigen/Dense>
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "node_def_builder.h"
#include "cpu_kernel_utils.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;
using namespace Eigen;

namespace {
const char *Test = "Cast";
}

class TEST_CAST_UT : public testing::Test {};

template <typename Tin, typename Tout>
void CalcExpectFunc(const NodeDef &node_def, Tin input_type, Tout expect_out[]) {
  auto input = node_def.MutableInputs(0);
  auto output = node_def.MutableOutputs(0);
  Tin *input_data = (Tin *)input->GetData();
  Tout *output_data = (Tout *)output->GetData();

  int64_t input_num = input->NumElements(); 

  for (int i = 0; i < input_num; i++) {
    expect_out[i] = (Tout)input_data[i];
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Cast", "Cast")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})           \
      .Attr("SrcT", data_types[0])                                 \
      .Attr("DstT", data_types[1]);

#define CAST_CASE_WITH_TYPE(base_type_in, aicpu_type_in, base_type_out,        \
                            aicpu_type_out)                                    \
  TEST_F(TEST_CAST_UT, TestCast_##aicpu_type_in##_To_##aicpu_type_out) {       \
    vector<DataType> data_types = {aicpu_type_in, aicpu_type_out};             \
    base_type_in input[6] = {(base_type_in)22, (base_type_in)32.3,             \
                             (base_type_in)-78.0, (base_type_in)-28.5,         \
                             (base_type_in)77, (base_type_in)0};               \
    base_type_out output[6] = {(base_type_out)0};                              \
    vector<void *> datas = {(void *)input, (void *)output};                    \
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};                         \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type_out expect_out[6] = {(base_type_out)0};                          \
    base_type_in input_type = (base_type_in)0;                                 \
    CalcExpectFunc(*node_def.get(), input_type, expect_out);                   \
    CompareResult<base_type_out>(output, expect_out, 6);                       \
  }          

CAST_CASE_WITH_TYPE(float, DT_FLOAT, int8_t, DT_INT8)

CAST_CASE_WITH_TYPE(float, DT_FLOAT, int16_t, DT_INT16)

CAST_CASE_WITH_TYPE(float, DT_FLOAT, int32_t, DT_INT32)

CAST_CASE_WITH_TYPE(float, DT_FLOAT, int64_t, DT_INT64)

CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int8_t, DT_INT8)

CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int16_t, DT_INT16)

CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int32_t, DT_INT32)

CAST_CASE_WITH_TYPE(double, DT_DOUBLE, int64_t, DT_INT64)