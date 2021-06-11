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

class TEST_ADAPTER_MAX_POOL2D_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, input, output0, output1, list_out)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "AdaptiveMaxPool2d", "AdaptiveMaxPool2d")                     \
      .Attr("output_size", list_out)           \
      .Input({"x", data_types[0], shapes[0], input})           \
      .Output({"y", data_types[1], shapes[1], output0})           \
      .Output({"argmax", data_types[2], shapes[1], output1});

#define CREATE_NODEDEF2(shapes, data_types, input, output0, output1, list_out)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "AdaptiveMaxPool2d", "AdaptiveMaxPool2d")                     \
      .Attr("output_size", list_out)           \
      .Input({"x", data_types[0], shapes[0], input})           \
      .Output({"y", data_types[1], shapes[1], output0})           \
      .Output({"argmax", data_types[2], shapes[2], output1});

#define ADPOOL2D_CASE_WITH_SHAPE(case_name, base_type, aicpu_type, out_base_type, out_aicpu_type, shapes, input, expect_output0, expect_output1, list_out)      \
  TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_##case_name) {                                  \
    int32_t out_data_num = sizeof(expect_output0)/sizeof(expect_output0[0]);           \
    vector<DataType> data_types = {aicpu_type, aicpu_type, out_aicpu_type};        \
    base_type output0[out_data_num] = {(base_type)0};                                \
    out_base_type output1[out_data_num] = {(out_base_type)0};                                          \
    CREATE_NODEDEF(shapes, data_types, input, output0, output1, list_out);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    EXPECT_EQ(CompareResult<base_type>(output0, expect_output0, out_data_num), true);                 \
    EXPECT_EQ(CompareResult<out_base_type>(output1, expect_output1, out_data_num), true);                 \
  }

#define ADPOOL2D_CASE_WITH_SHAPE_DISMATCH(case_name, base_type, aicpu_type, out_base_type, out_aicpu_type, shapes, input, expect_output0, expect_output1, list_out)      \
  TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_##case_name) {                                  \
    int32_t out_data_num = sizeof(expect_output0)/sizeof(expect_output0[0]);           \
    vector<DataType> data_types = {aicpu_type, aicpu_type, out_aicpu_type};        \
    base_type output0[out_data_num] = {(base_type)0};                                \
    out_base_type output1[out_data_num] = {(out_base_type)0};                                          \
    CREATE_NODEDEF(shapes, data_types, input, output0, output1, list_out);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                              \
  }

#define ADPOOL2D_CASE_WITH_SHAPE_DISMATCH2(case_name, base_type, aicpu_type, out_base_type, out_aicpu_type, shapes, input, expect_output0, expect_output1, list_out)      \
  TEST_F(TEST_ADAPTER_MAX_POOL2D_UT, TestAdaptiveMaxPool2d_##case_name) {                                  \
    int32_t out_data_num = sizeof(expect_output0)/sizeof(expect_output0[0]);           \
    vector<DataType> data_types = {aicpu_type, aicpu_type, out_aicpu_type};        \
    base_type output0[out_data_num] = {(base_type)0};                                \
    out_base_type output1[out_data_num] = {(out_base_type)0};                                          \
    CREATE_NODEDEF2(shapes, data_types, input, output0, output1, list_out);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                              \
  }



vector<int64_t> list_out_1 = {1, 2};
vector<vector<int64_t>> shapes_1 = {{2, 2, 3}, {2, 1, 2}};
float_t input_1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
float_t expect_output0_1[] = {5, 6, 11, 12};
int32_t expect_output1_1[] = {4, 5, 4, 5};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_1, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_1, input_1, expect_output0_1,
                          expect_output1_1, list_out_1)


vector<int64_t> list_out_2 = {2, 2};
vector<vector<int64_t>> shapes_2 = {{1, 4, 4}, {1, 2, 2}};
float_t input_2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_2[] = {6, 8, 14, 16};
int32_t expect_output1_2[] = {5, 7, 13, 15};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_2, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_2, input_2, expect_output0_2,
                          expect_output1_2, list_out_2)


vector<int64_t> list_out_3 = {2, 1};
vector<vector<int64_t>> shapes_3 = {{4, 2, 3}, {4, 2, 1}};
float_t input_3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
float_t expect_output0_3[] = {3,6,9,12,15,18,21,24};
int64_t expect_output1_3[] = {2,5,2,5,2,5,2,5};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_3, float_t, DT_FLOAT, int64_t, DT_INT64, shapes_3, input_3, expect_output0_3,
                          expect_output1_3, list_out_3)

vector<int64_t> list_out_4 = {3, 4};
vector<vector<int64_t>> shapes_4 = {{1, 1, 4}, {1, 3, 4}};
float_t input_4[] = {1, 2, 3, 4};
float_t expect_output0_4[] = {1,2,3,4,1,2,3,4,1,2,3,4};
int64_t expect_output1_4[] = {0,1,2,3,0,1,2,3,0,1,2,3};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_4, float_t, DT_FLOAT, int64_t, DT_INT64, shapes_4, input_4, expect_output0_4,
                          expect_output1_4, list_out_4)

vector<int64_t> list_out_5 = {3, 4};
vector<vector<int64_t>> shapes_5 = {{1, 2, 2}, {1, 3, 4}};
float_t input_5[] = {1, 2, 3, 4};
float_t expect_output0_5[] = {1,1,2,2,3,3,4,4,3,3,4,4};
int64_t expect_output1_5[] = {0,0,1,1,2,2,3,3,2,2,3,3};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_5, float_t, DT_FLOAT, int64_t, DT_INT64, shapes_5, input_5, expect_output0_5,
                          expect_output1_5, list_out_5)

vector<int64_t> list_out_6 = {2};
vector<vector<int64_t>> shapes_6 = {{1, 4, 4}, {1, 2, 2}};
float_t input_6[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_6[] = {6, 8, 14, 16};
int32_t expect_output1_6[] = {5, 7, 13, 15};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_6, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_6, input_6, expect_output0_6,
                          expect_output1_6, list_out_6)

vector<int64_t> list_out_7 = {1,2};
vector<vector<int64_t>> shapes_7 = {{2,1,2,4}, {2,1,1,2}};
float_t input_7[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_7[] = {6, 8, 14, 16};
int32_t expect_output1_7[] = {5, 7, 5, 7};
ADPOOL2D_CASE_WITH_SHAPE(dapter_max_pool2d_float_succ_7, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_7, input_7, expect_output0_7,
                          expect_output1_7, list_out_7)

vector<int64_t> list_out_8 = {2};
vector<vector<int64_t>> shapes_8 = {{1, 4, 4}, {1, 2, 2}};
float_t input_8[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_8[] = {6, 8, 14, 16};
int32_t expect_output1_8[] = {5, 7, 13, 15};
ADPOOL2D_CASE_WITH_SHAPE_DISMATCH(dapter_max_pool2d_float_failed_1, float_t, DT_FLOAT, int8_t, DT_INT8, shapes_8, input_8, expect_output0_8,
                          expect_output1_8, list_out_8)

vector<int64_t> list_out_9 = {2};
vector<vector<int64_t>> shapes_9 = {{1, 4, 4}, {1, 2, 2}};
float_t expect_output0_9[] = {6, 8, 14, 16};
int32_t expect_output1_9[] = {5, 7, 13, 15};
ADPOOL2D_CASE_WITH_SHAPE_DISMATCH(dapter_max_pool2d_float_failed_2, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_9, nullptr, expect_output0_9,
                          expect_output1_9, list_out_9)

vector<vector<int64_t>> shapes_10 = {{1, 4, 4}, {1, 2, 2}};
float_t input_10[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_10[] = {6, 8, 14, 16};
int32_t expect_output1_10[] = {5, 7, 13, 15};
ADPOOL2D_CASE_WITH_SHAPE_DISMATCH(dapter_max_pool2d_float_failed_3, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_10, input_10, expect_output0_10,
                          expect_output1_10, nullptr)

vector<int64_t> list_out_11 = {2};
vector<vector<int64_t>> shapes_11 = {{1,1,1, 4, 4}, {1, 2, 2}};
float_t input_11[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float_t expect_output0_11[] = {6, 8, 14, 16};
int32_t expect_output1_11[] = {5, 7, 13, 15};
ADPOOL2D_CASE_WITH_SHAPE_DISMATCH(dapter_max_pool2d_float_failed_4, float_t, DT_FLOAT, int32_t, DT_INT32, shapes_11, input_11, expect_output0_11,
                          expect_output1_11, list_out_11)

vector<int64_t> list_out_12 = {3, 4};
vector<vector<int64_t>> shapes_12 = {{1, 1, 4}, {1, 1, 1}, {1, 3, 4}};
float_t input_12[] = {1, 2, 3, 4};
float_t expect_output0_12[] = {1,2,3,4,1,2,3,4,1,2,3,4};
int64_t expect_output1_12[] = {0,1,2,3,0,1,2,3,0,1,2,3};
ADPOOL2D_CASE_WITH_SHAPE_DISMATCH2(dapter_max_pool2d_float_shape_failed_1, float_t, DT_FLOAT, int64_t, DT_INT64, shapes_12, input_12, expect_output0_12,
                          expect_output1_12, list_out_12)

vector<int64_t> list_out_13 = {3, 4};
vector<vector<int64_t>> shapes_13 = {{1, 1, 4}, {1, 3, 4}, {1, 1, 1}};
float_t input_13[] = {1, 2, 3, 4};
float_t expect_output0_13[] = {1,2,3,4,1,2,3,4,1,2,3,4};
int64_t expect_output1_13[] = {0,1,2,3,0,1,2,3,0,1,2,3};
ADPOOL2D_CASE_WITH_SHAPE_DISMATCH2(dapter_max_pool2d_float_shape_failed_2, float_t, DT_FLOAT, int64_t, DT_INT64, shapes_13, input_13, expect_output0_13,
                          expect_output1_13, list_out_13)


