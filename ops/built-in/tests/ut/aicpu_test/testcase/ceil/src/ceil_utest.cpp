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
#include <math.h>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_CEIL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                              \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "Ceil", "Ceil")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                        \
      .Output({"y", data_types[1], shapes[1], datas[1]})

#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_CEIL_UT, TestTopK_##aicpu_type) {                                \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                    \
    vector<vector<int64_t>> shapes = {{24}, {24}};                             \
    base_type input[24];                                                       \
    SetRandomValue<base_type>(input, 24);                                      \
    base_type output[24] = {(base_type)0};                                     \
    vector<void *> datas = {(void *)input, (void *)output};                    \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
  }

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)

ADD_CASE(double, DT_DOUBLE)

