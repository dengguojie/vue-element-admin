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


TEST_F(TEST_WHERE_UT, TestWhere_bool_where) {
    vector<DataType> data_types = {DT_BOOL, DT_INT64};
    bool input[data_nums[0]] = {1, 1, 1, 0};
    int64_t output[data_nums[1]] = {0};
    vector<void *> datas = {(void *)input, (void *)output};
    CREATE_NODEDEF(where_shapes1, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int64_t expect_out[data_nums[1]] = {0, 0, 0, 1, 1, 0};
    bool res = CompareResult<int64_t>(output, expect_out, data_nums[1]);
    EXPECT_EQ(res, true);
}

TEST_F(TEST_WHERE_UT, TestWhere_int8_where) {
    vector<DataType> data_types = {DT_INT8, DT_INT64};
    int8_t input[data_nums[0]] = {1, 1, 1, 0};
    int64_t output[data_nums[1]] = {0};
    vector<void *> datas = {(void *)input, (void *)output};
    CREATE_NODEDEF(where_shapes1, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int64_t expect_out[data_nums[1]] = {0, 0, 0, 1, 1, 0};
    bool res = CompareResult<int64_t>(output, expect_out, data_nums[1]);
    EXPECT_EQ(res, true);
}

TEST_F(TEST_WHERE_UT, TestWhere_WITH_SHAPE_int8_1D) {
    vector<DataType> data_types = {DT_INT8, DT_INT64};
    vector<int8_t> input(data.begin(), data.end());
    int64_t output[40] = {0};
    vector<void *> datas = {(void *)input.data(), (void *)output};
    CREATE_NODEDEF(shape2, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int nums = shape2[1][0] * shape2[1][1];
    bool res = CompareResult<int64_t>(output, shape2_res.data(), nums);
    EXPECT_EQ(res, true);
}

TEST_F(TEST_WHERE_UT, ADD_EMPTY_CASE) {
    vector<DataType> data_types = {DT_INT8, DT_INT64};
    int64_t output[40] = {0};
    int64_t expect_out[1] = {0};
    vector<void *> datas = {nullptr, (void *)output} ;
    CREATE_NODEDEF(shape1, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool res = CompareResult<int64_t>(output, expect_out, 0);
    EXPECT_EQ(res, true);
}
