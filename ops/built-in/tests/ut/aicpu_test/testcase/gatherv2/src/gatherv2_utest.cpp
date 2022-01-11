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

class TEST_GATHERV2_UT : public testing::Test {};

#define CREATE_GATHERV2_NODEDEF(shapes, data_types, datas)          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();  \
  NodeDefBuilder(node_def.get(), "GatherV2", "GatherV2")            \
      .Input({"x", data_types[0], shapes[0], datas[0]})             \
      .Input({"indices", data_types[1], shapes[1], datas[1]})       \
      .Input({"axis", data_types[2], shapes[2], datas[2]})          \
      .Output({"output", data_types[3], shapes[3], datas[3]})       \
      .Attr("batch_dims", 0);

#define ADD_GATHERV2_CASE(base_type, aicpu_type, indices_type, indices_aicpu_type)                  \
  TEST_F(TEST_GATHERV2_UT, TestGatherV2_##aicpu_type) {                                             \
    vector<DataType> data_types = {aicpu_type, indices_aicpu_type, DT_INT64, aicpu_type};           \
    vector<vector<int64_t>> shapes = {{3, 5}, {2}, {1}, {3, 2}};                                          \
    base_type input0[15] = {(base_type)0, (base_type)1, (base_type)2, (base_type)3, (base_type)4,   \
                            (base_type)6, (base_type)1, (base_type)8, (base_type)3, (base_type)4,  \
                            (base_type)7, (base_type)1, (base_type)9, (base_type)3, (base_type)4};   \
    indices_type input1[2] = {(indices_type)0, (indices_type)2};                                                     \
    int64_t input2[1] = {1};                                                                        \
    base_type output[6] = {(base_type)0};      \
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output};        \
    CREATE_GATHERV2_NODEDEF(shapes, data_types, datas);                      \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    base_type expect_out[6] = {(base_type)0, (base_type)2, (base_type)6, (base_type)8, (base_type)7, (base_type)9};      \
    EXPECT_EQ(CompareResult<base_type>(output, expect_out, 6), true);    \
  }

  ADD_GATHERV2_CASE(int64_t, DT_INT64, int64_t, DT_INT64)
  ADD_GATHERV2_CASE(int32_t, DT_INT32, int64_t, DT_INT64)

TEST_F(TEST_GATHERV2_UT, TestGatherV2_HighRank) {
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{4}, {2, 3}, {1}, {2, 3}};
    int64_t input0[4] = {0, 1, 2, 3};
    int64_t input1[6] = {1, 2, 0, 2, 3, 0};
    int64_t input2[1] = {0};
    int64_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output};
    CREATE_GATHERV2_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int64_t expect_out[6] = {1, 2, 0, 2, 3, 0};
    EXPECT_EQ(CompareResult<int64_t>(output, expect_out, 6), true);
}

TEST_F(TEST_GATHERV2_UT, TestGatherV2_Complex) {
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT64, DT_INT64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{5}, {1}, {1}, {1}};
    std::complex<float> input0[5] = {std::complex<float>(0, 10), std::complex<float>(1, 11),
                                     std::complex<float>(2, 12), std::complex<float>(3, 13),
                                     std::complex<float>(4, 14)};
    int64_t input1[1] = {3};
    int64_t input2[1] = {0};
    std::complex<float> output[1] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output};
    CREATE_GATHERV2_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    std::complex<float> expect_out[1] = {std::complex<float>(3, 13)};
    EXPECT_EQ(CompareResult<std::complex<float>>(output, expect_out, 1), true);
}

TEST_F(TEST_GATHERV2_UT, TestGatherV2_Float32) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_INT64, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{4}, {2, 3}, {1}, {2, 3}};
    float input0[4] = {0.01051331, 2.1171875, 6.3320312, 7.7382812};
    int64_t input1[6] = {1, 2, 0, 3};
    int64_t input2[1] = {0};
    float output[4] = {0.0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)output};
    CREATE_GATHERV2_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[4] = {2.1171875, 6.3320312, 0.01051331, 7.7382812};
    EXPECT_EQ(CompareResult<float>(output, expect_out, 4), true);
}