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

class TEST_WHERE_UT : public testing::Test {};


#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Where", "Where")                 \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]});

#define ADD_CASE(case_name, base_type, aicpu_type, shape, data_num, input_data)\
  TEST_F(TEST_WHERE_UT, TestWhere_##case_name) {                               \
    vector<DataType> data_types = {aicpu_type, DT_INT64};                      \
    base_type input[data_num[0]] = {1, 1, 1, 0};                               \
    int64_t output[data_num[1]] = {0};                                         \
    vector<void *> datas = {(void *)input, (void *)output};                    \
    CREATE_NODEDEF(shape, data_types, datas);                                  \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    int64_t expect_out[data_num[1]] = {0, 0, 0, 1, 1, 0};                      \
    bool res = CompareResult<int64_t>(output, expect_out, data_num[1]);        \
    EXPECT_EQ(res, true);                                                      \
  }

#define ADD_CASE_WITH_SHAPE(case_name, data_type, aicpu_type, shapes, data,    \
                            expect_out)                                        \
  TEST_F(TEST_WHERE_UT, TestWhere_WITH_SHAPE_##case_name) {                    \
    vector<DataType> data_types = {aicpu_type, DT_INT64};                      \
    vector<data_type> input(data.begin(), data.end());                         \
    int64_t output[40] = {0};                                                  \
    vector<void *> datas = {(void *)input.data(), (void *)output};             \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    int nums = shapes[1][0] * shapes[1][1];                                    \
    bool res = CompareResult<int64_t>(output, expect_out.data(), nums);        \
    EXPECT_EQ(res, true);                                                      \
  }

#define ADD_EMPTY_CASE(data_type, aicpu_type, shapes)                          \
  TEST_F(TEST_WHERE_UT, ADD_EMPTY_CASE) {                                      \
    vector<DataType> data_types = {aicpu_type, DT_INT64};                      \
    int64_t output[40] = {0};                                                  \
    int64_t expect_out[1] = {0};                                               \
    vector<void *> datas = {nullptr, (void *)output} ;                         \
    CREATE_NODEDEF(shapes, data_types, datas);                                 \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    bool res = CompareResult<int64_t>(output, expect_out, 0);                  \
    EXPECT_EQ(res, true);                                                      \
  }


vector<int> data{0, 1, 2, 0, 4, 0, 3, 7};
vector<int> data_0{0};
vector<int64_t> data_nums = {4, 6};
vector<vector<int64_t>> where_shapes1 = {{2, 2}, {3, 2}};
vector<vector<int64_t>> shape1 = {{0}, {0, 1}};
vector<int64_t> shape1_res = {};
vector<vector<int64_t>> shape2 = {{8}, {5, 1}};
vector<int64_t> shape2_res = {1, 2, 4, 6, 7};
vector<vector<int64_t>> shape3 = {{2, 4}, {5, 2}};
vector<int64_t> shape3_res = {0, 1, 0, 2, 1, 0, 1, 2, 1, 3};
vector<vector<int64_t>> shape4 = {{2, 2, 2}, {5, 3}};
vector<int64_t> shape4_res = {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1};
vector<vector<int64_t>> shape5 = {{2, 1, 2, 2}, {5, 4}};
vector<int64_t> shape5_res = {0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 
                              0, 1, 0, 1, 1};
vector<vector<int64_t>> shape6 = {{2, 1, 2, 1, 2}, {5, 5}};
vector<int64_t> shape6_res = {0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                              1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                              1, 0, 1, 0, 1};
vector<vector<int64_t>> shape7 = {{2, 1, 1, 2, 1, 2}, {5, 6}};
vector<int64_t> shape7_res = {0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                              1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                              1, 0, 0, 1, 0, 1};
vector<vector<int64_t>> shape8 = {{2, 1, 1, 2, 1, 1, 2}, {5, 7}};
vector<int64_t> shape8_res = {0, 0, 0, 0, 0, 0, 1,
                              0, 0, 0, 1, 0, 0, 0,
                              1, 0, 0, 0, 0, 0, 0,
                              1, 0, 0, 1, 0, 0, 0,
                              1, 0, 0, 1, 0, 0, 1};
vector<vector<int64_t>> shape9 = {{2, 1, 1, 2, 1, 1, 2, 1}, {5, 8}};
vector<int64_t> shape9_res = {0, 0, 0, 0, 0, 0, 1, 0,
                              0, 0, 0, 1, 0, 0, 0, 0,
                              1, 0, 0, 0, 0, 0, 0, 0,
                              1, 0, 0, 1, 0, 0, 0, 0,
                              1, 0, 0, 1, 0, 0, 1, 0};
ADD_CASE(bool_where, bool, DT_BOOL, where_shapes1, data_nums, data)
ADD_CASE(int8_where, int8_t, DT_INT8, where_shapes1, data_nums, data)
ADD_CASE(int16_where, int16_t, DT_INT16, where_shapes1, data_nums, data)
ADD_CASE(int32_where, int32_t, DT_INT32, where_shapes1, data_nums, data)
ADD_CASE(int64_where, int64_t, DT_INT64, where_shapes1, data_nums, data)
ADD_CASE(uint8_where, uint8_t, DT_UINT8, where_shapes1, data_nums, data)
ADD_CASE(uint16_where, uint16_t, DT_UINT16, where_shapes1, data_nums, data)
ADD_CASE(uint32_where, uint32_t, DT_UINT32, where_shapes1, data_nums, data)
ADD_CASE(uint64_where, uint64_t, DT_UINT64, where_shapes1, data_nums, data)
ADD_CASE(float_where, float, DT_FLOAT, where_shapes1, data_nums, data)
ADD_CASE(double_where, double, DT_DOUBLE, where_shapes1, data_nums, data)

ADD_EMPTY_CASE(int8_t, DT_INT8, shape1)
ADD_CASE_WITH_SHAPE(int8_1D, int8_t, DT_INT8, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(int8_2D, int8_t, DT_INT8, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(int8_3D, int8_t, DT_INT8, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(int8_4D, int8_t, DT_INT8, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(int8_5D, int8_t, DT_INT8, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(int8_6D, int8_t, DT_INT8, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(int8_7D, int8_t, DT_INT8, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(int8_8D, int8_t, DT_INT8, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(int16_1D, int16_t, DT_INT16, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(int16_2D, int16_t, DT_INT16, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(int16_3D, int16_t, DT_INT16, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(int16_4D, int16_t, DT_INT16, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(int16_5D, int16_t, DT_INT16, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(int16_6D, int16_t, DT_INT16, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(int16_7D, int16_t, DT_INT16, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(int16_8D, int16_t, DT_INT16, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(int32_1D, int32_t, DT_INT32, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(int32_2D, int32_t, DT_INT32, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(int32_3D, int32_t, DT_INT32, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(int32_4D, int32_t, DT_INT32, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(int32_5D, int32_t, DT_INT32, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(int32_6D, int32_t, DT_INT32, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(int32_7D, int32_t, DT_INT32, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(int32_8D, int32_t, DT_INT32, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(int64_1D, int64_t, DT_INT64, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(int64_2D, int64_t, DT_INT64, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(int64_3D, int64_t, DT_INT64, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(int64_4D, int64_t, DT_INT64, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(int64_5D, int64_t, DT_INT64, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(int64_6D, int64_t, DT_INT64, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(int64_7D, int64_t, DT_INT64, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(int64_8D, int64_t, DT_INT64, shape9, data, shape9_res)


ADD_CASE_WITH_SHAPE(uint8_1D, uint8_t, DT_UINT8, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(uint8_2D, uint8_t, DT_UINT8, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(uint8_3D, uint8_t, DT_UINT8, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(uint8_4D, uint8_t, DT_UINT8, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(uint8_5D, uint8_t, DT_UINT8, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(uint8_6D, uint8_t, DT_UINT8, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(uint8_7D, uint8_t, DT_UINT8, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(uint8_8D, uint8_t, DT_UINT8, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(uint16_1D, uint16_t, DT_UINT16, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(uint16_2D, uint16_t, DT_UINT16, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(uint16_3D, uint16_t, DT_UINT16, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(uint16_4D, uint16_t, DT_UINT16, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(uint16_5D, uint16_t, DT_UINT16, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(uint16_6D, uint16_t, DT_UINT16, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(uint16_7D, uint16_t, DT_UINT16, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(uint16_8D, uint16_t, DT_UINT16, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(uint32_1D, uint32_t, DT_UINT32, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(uint32_2D, uint32_t, DT_UINT32, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(uint32_3D, uint32_t, DT_UINT32, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(uint32_4D, uint32_t, DT_UINT32, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(uint32_5D, uint32_t, DT_UINT32, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(uint32_6D, uint32_t, DT_UINT32, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(uint32_7D, uint32_t, DT_UINT32, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(uint32_8D, uint32_t, DT_UINT32, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(uint64_1D, uint64_t, DT_UINT64, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(uint64_2D, uint64_t, DT_UINT64, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(uint64_3D, uint64_t, DT_UINT64, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(uint64_4D, uint64_t, DT_UINT64, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(uint64_5D, uint64_t, DT_UINT64, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(uint64_6D, uint64_t, DT_UINT64, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(uint64_7D, uint64_t, DT_UINT64, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(uint64_8D, uint64_t, DT_UINT64, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(float_1D, float, DT_FLOAT, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(float_2D, float, DT_FLOAT, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(float_3D, float, DT_FLOAT, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(float_4D, float, DT_FLOAT, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(float_5D, float, DT_FLOAT, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(float_6D, float, DT_FLOAT, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(float_7D, float, DT_FLOAT, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(float_8D, float, DT_FLOAT, shape9, data, shape9_res)

ADD_CASE_WITH_SHAPE(double_1D, double, DT_DOUBLE, shape2, data, shape2_res)
ADD_CASE_WITH_SHAPE(double_2D, double, DT_DOUBLE, shape3, data, shape3_res)
ADD_CASE_WITH_SHAPE(double_3D, double, DT_DOUBLE, shape4, data, shape4_res)
ADD_CASE_WITH_SHAPE(double_4D, double, DT_DOUBLE, shape5, data, shape5_res)
ADD_CASE_WITH_SHAPE(double_5D, double, DT_DOUBLE, shape6, data, shape6_res)
ADD_CASE_WITH_SHAPE(double_6D, double, DT_DOUBLE, shape7, data, shape7_res)
ADD_CASE_WITH_SHAPE(double_7D, double, DT_DOUBLE, shape8, data, shape8_res)
ADD_CASE_WITH_SHAPE(double_8D, double, DT_DOUBLE, shape9, data, shape9_res)
