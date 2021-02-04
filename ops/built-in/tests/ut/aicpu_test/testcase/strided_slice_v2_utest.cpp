#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include <cmath>

#include "Eigen/Core"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_STRIDED_SLICE_V2_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Input({"axes", data_types[3], shapes[3], datas[3]})           \
      .Input({"strides", data_types[4], shapes[4], datas[4]})        \
      .Output({"y", data_types[5], shapes[5], datas[5]});

#define CREATE_NODEDEF2(shapes, data_types, datas)                   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Output({"y", data_types[3], shapes[3], datas[3]});

#define STRIDED_SLICE_V2_CASE1(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_##aicpu_type) {   \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {1, 1, 3}};                      \
    base_type x[3 * 2 * 3];                                             \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 1;                                                       \
    begin[1] = 0;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 2;                                                         \
    end[1] = 1;                                                         \
    end[2] = 3;                                                         \
                                                                        \
    int32_t axes[3];                                                    \
    axes[0] = 0;                                                        \
    axes[1] = 1;                                                        \
    axes[2] = 2;                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 1;                                                     \
    strides[1] = 1;                                                     \
    strides[2] = 1;                                                     \
                                                                        \
    base_type y[1 * 1 * 3] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[1 * 1 * 3] = {(base_type)0};                   \
    expect_out[0] = (base_type)6;                                       \
    expect_out[1] = (base_type)7;                                       \
    expect_out[2] = (base_type)8;                                       \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 3), true);        \
  }

STRIDED_SLICE_V2_CASE1(Eigen::half, DT_FLOAT16)
STRIDED_SLICE_V2_CASE1(float, DT_FLOAT)
STRIDED_SLICE_V2_CASE1(double, DT_DOUBLE)
STRIDED_SLICE_V2_CASE1(int8_t, DT_INT8)
STRIDED_SLICE_V2_CASE1(int16_t, DT_INT16)
STRIDED_SLICE_V2_CASE1(int32_t, DT_INT32)
STRIDED_SLICE_V2_CASE1(int64_t, DT_INT64)
STRIDED_SLICE_V2_CASE1(uint8_t, DT_UINT8)
STRIDED_SLICE_V2_CASE1(uint16_t, DT_UINT16)
STRIDED_SLICE_V2_CASE1(uint32_t, DT_UINT32)
STRIDED_SLICE_V2_CASE1(uint64_t, DT_UINT64)

#define STRIDED_SLICE_V2_CASE2(base_type, aicpu_type)                    \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_2_##aicpu_type) {  \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,       \
                                   DT_INT32,   DT_INT32, aicpu_type};    \
    vector<vector<int64_t>> shapes = {                                   \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {2, 1, 1}};                       \
    base_type x[3 * 2 * 3];                                              \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                                \
      x[i] = (base_type)i;                                               \
    }                                                                    \
                                                                         \
    int32_t begin[3];                                                    \
    begin[0] = 1;                                                        \
    begin[1] = 0;                                                        \
    begin[2] = 0;                                                        \
                                                                         \
    int32_t end[3];                                                      \
    end[0] = 2;                                                          \
    end[1] = 1;                                                          \
    end[2] = 3;                                                          \
                                                                         \
    int32_t axes[3];                                                     \
    axes[0] = 1;                                                         \
    axes[1] = 2;                                                         \
    axes[2] = 0;                                                         \
                                                                         \
    int32_t strides[3];                                                  \
    strides[0] = 1;                                                      \
    strides[1] = 1;                                                      \
    strides[2] = 2;                                                      \
                                                                         \
    base_type y[2 * 1 * 1] = {(base_type)0};                             \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end,  \
                            (void *)axes, (void *)strides, (void *)y};   \
    CREATE_NODEDEF(shapes, data_types, datas);                           \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                        \
    base_type expect_out[2 * 1 * 1] = {(base_type)0};                    \
    expect_out[0] = (base_type)3;                                        \
    expect_out[1] = (base_type)15;                                       \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 2 * 1 * 1), true); \
  }

STRIDED_SLICE_V2_CASE2(int32_t, DT_INT32)

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();   \
  NodeDefBuilder(node_def.get(), "StridedSliceV2", "StridedSliceV2") \
      .Input({"x", data_types[0], shapes[0], datas[0]})              \
      .Input({"begin", data_types[1], shapes[1], datas[1]})          \
      .Input({"end", data_types[2], shapes[2], datas[2]})            \
      .Input({"axes", data_types[3], shapes[3], datas[3]})           \
      .Input({"strides", data_types[4], shapes[4], datas[4]})        \
      .Output({"y", data_types[5], shapes[5], datas[5]});

#define STRIDED_SLICE_V2_CASE3(base_type, aicpu_type)                          \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_3_##aicpu_type) {        \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,             \
                                   aicpu_type};                                \
    vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3}, {1, 1, 3}};         \
    base_type x[3 * 2 * 3];                                                    \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                                      \
      x[i] = (base_type)i;                                                     \
    }                                                                          \
                                                                               \
    int32_t begin[3];                                                          \
    begin[0] = 1;                                                              \
    begin[1] = 0;                                                              \
    begin[2] = 0;                                                              \
                                                                               \
    int32_t end[3];                                                            \
    end[0] = 2;                                                                \
    end[1] = 1;                                                                \
    end[2] = 3;                                                                \
                                                                               \
    base_type y[1 * 1 * 3] = {(base_type)0};                                   \
    vector<void *> datas = {(void *)x, (void *)begin, (void *)end, (void *)y}; \
    CREATE_NODEDEF2(shapes, data_types, datas);                                \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[1 * 1 * 3] = {(base_type)0};                          \
    expect_out[0] = (base_type)6;                                              \
    expect_out[1] = (base_type)7;                                              \
    expect_out[2] = (base_type)8;                                              \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 3), true);               \
  }

STRIDED_SLICE_V2_CASE3(int32_t, DT_INT32)

#define STRIDED_SLICE_V2_CASE4(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_4_##aicpu_type) { \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {1, 2, 3}};                      \
    base_type x[3 * 2 * 3];                                             \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 1;                                                       \
    begin[1] = 0;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 2;                                                         \
    end[1] = 2;                                                         \
    end[2] = 3;                                                         \
                                                                        \
    int32_t axes[3];                                                    \
    axes[0] = 0;                                                        \
    axes[1] = 1;                                                        \
    axes[2] = 2;                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 1;                                                     \
    strides[1] = 1;                                                     \
    strides[2] = 1;                                                     \
                                                                        \
    base_type y[1 * 2 * 3] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[1 * 2 * 3] = {(base_type)0};                   \
    expect_out[0] = (base_type)6;                                       \
    expect_out[1] = (base_type)7;                                       \
    expect_out[2] = (base_type)8;                                       \
    expect_out[3] = (base_type)9;                                       \
    expect_out[4] = (base_type)10;                                      \
    expect_out[5] = (base_type)11;                                      \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 6), true);        \
  }

STRIDED_SLICE_V2_CASE4(int64_t, DT_INT64)

#define STRIDED_SLICE_V2_CASE5(base_type, aicpu_type)                   \
  TEST_F(TEST_STRIDED_SLICE_V2_UT, TestStridedSliceV2_5_##aicpu_type) { \
    vector<DataType> data_types = {aicpu_type, DT_INT32, DT_INT32,      \
                                   DT_INT32,   DT_INT32, aicpu_type};   \
    vector<vector<int64_t>> shapes = {                                  \
        {3, 2, 3}, {3}, {3}, {3}, {3}, {2, 1, 2}};                      \
    base_type x[3 * 2 * 3];                                             \
    for (int i = 0; i < 3 * 2 * 3; ++i) {                               \
      x[i] = (base_type)i;                                              \
    }                                                                   \
                                                                        \
    int32_t begin[3];                                                   \
    begin[0] = 0;                                                       \
    begin[1] = 1;                                                       \
    begin[2] = 0;                                                       \
                                                                        \
    int32_t end[3];                                                     \
    end[0] = 3;                                                         \
    end[1] = 2;                                                         \
    end[2] = 3;                                                         \
                                                                        \
    int32_t axes[3];                                                    \
    axes[0] = 0;                                                        \
    axes[1] = 1;                                                        \
    axes[2] = 2;                                                        \
                                                                        \
    int32_t strides[3];                                                 \
    strides[0] = 2;                                                     \
    strides[1] = 1;                                                     \
    strides[2] = 2;                                                     \
                                                                        \
    base_type y[2 * 1 * 2] = {(base_type)0};                            \
    vector<void *> datas = {(void *)x,    (void *)begin,   (void *)end, \
                            (void *)axes, (void *)strides, (void *)y};  \
    CREATE_NODEDEF(shapes, data_types, datas);                          \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                       \
    base_type expect_out[2 * 1 * 2] = {(base_type)0};                   \
    expect_out[0] = (base_type)3;                                       \
    expect_out[1] = (base_type)5;                                       \
    expect_out[2] = (base_type)15;                                      \
    expect_out[3] = (base_type)17;                                      \
    EXPECT_EQ(CompareResult<base_type>(y, expect_out, 4), true);        \
  }

STRIDED_SLICE_V2_CASE5(float, DT_FLOAT)
