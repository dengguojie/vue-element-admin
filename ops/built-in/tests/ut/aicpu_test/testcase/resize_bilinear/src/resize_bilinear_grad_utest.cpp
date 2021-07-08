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

class TEST_ResizeBilinearGrad_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();           \
  NodeDefBuilder(node_def.get(), "ResizeBilinearGrad", "ResizeBilinearGrad") \
      .Input({"dy", data_types[0], shapes[0], datas[0]})                     \
      .Input({"x", data_types[1], shapes[1], datas[1]})                      \
      .Output({"dx", data_types[2], shapes[2], datas[2]})                    \
      .Attr("align_corners", false)

#define ADD_CASE(base_type, aicpu_type)                                                                          \
  TEST_F(TEST_ResizeBilinearGrad_UT, TestResizeBilinearGrad_##aicpu_type) {                                      \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};                                          \
    vector<vector<int64_t>> shapes = {{1, 1, 2, 4}, {1, 1, 2, 5}, {1, 1, 2, 5}};                                 \
    base_type dy[2 * 4] = {base_type(1), base_type(2), base_type(3), base_type(4),                               \
                           base_type(2), base_type(4), base_type(6), base_type(4)};                              \
    base_type x[2 * 5] = {base_type(1), base_type(2), base_type(3), base_type(4), base_type(5),                  \
                          base_type(2), base_type(4), base_type(6), base_type(4), base_type(9)};                 \
    base_type output[2 * 5];                                                                                     \
    base_type output_expect[2 * 5] = {base_type(1), base_type(1.5), base_type(2), base_type(2.5), base_type(3),  \
                                      base_type(2), base_type(3),   base_type(4), base_type(4),   base_type(3)}; \
    vector<void*> datas = {(void*)dy, (void*)x, (void*)output};                                                  \
    CREATE_NODEDEF(shapes, data_types, datas);                                                                   \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                                                                \
    EXPECT_EQ(0, std::memcmp(output, output_expect, sizeof(output)));                                            \
  }

ADD_CASE(Eigen::half, DT_FLOAT16)

ADD_CASE(float, DT_FLOAT)