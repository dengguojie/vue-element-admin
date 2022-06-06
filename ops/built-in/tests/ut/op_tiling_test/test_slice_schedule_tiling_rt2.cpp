//
// Created by wangyu on 2022/5/17.
//
#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/slice_dsl.h"

#include "common_autotiling_util.h"
#include "graph/utils/attr_utils.h"

#define private public
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace optiling;


class SliceScheduleTilingRt2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SliceScheduleTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SliceScheduleTiling TearDown" << std::endl;
  }
};

TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_0
) {
// slice static lr depad
vector<vector<int64_t>> inputs {
    {32509728, 25},
};
vector<vector<int64_t>> outputs{
    {32509728, 20},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 65280;
op_compile_info.x_type_size = 4;
op_compile_info.x_align = 8;
op_compile_info.is_static = true;

op_compile_info.size = {32509728, 20};
op_compile_info.begin = {0, 0};
op_compile_info.end_mode = 0;
op_compile_info.is_const_sizes = true;
op_compile_info.is_const_ends = false;
op_compile_info.is_const_begins = true;
op_compile_info.coex_list = {2, 4, 3};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "527000000, 0, 1015929, 0, 504, 7, 32";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_1
) {
// TODO const input condtion
// slice data mov
vector<vector<int64_t>> inputs {
    {512, 50, 1324},
    {3,},
    {3,}
};
vector<vector<int64_t>> outputs{
    {32509728, 20},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 65280;
op_compile_info.x_type_size = 4;
op_compile_info.x_align = 8;
op_compile_info.is_static = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.coex_list = {2, 4, 3};
op_compile_info.end_mode = 0;

op_compile_info.slice_vars = {
    {"521000000", {10000, 10001, 20000, 30000, 20001, 30001, 40000, 50000}},
};
op_compile_info.is_valid = true;
//vector<int32_t> begin_data{0, 0, 0};
//vector<int32_t> size_data{-1, -1, 512};
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[3] = {0, 0, 0};
test.SetInt32ConstInput(1, begin_data, 3);
int32_t size_data[3] = {-1, -1, 512};
test.SetInt32ConstInput(2, size_data, 3);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "25600, 1324, 0, 25600, 0, 512, 800, 112";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_2
) {
// TODO const input condtion
// slice one dim
vector<vector<int64_t>> inputs {
    {3, 20},
    {2},
    {2}
};
vector<vector<int64_t>> outputs{
    {1, 20},
};
ge::DataType dtype = ge::DT_INT32;


SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 65280;
op_compile_info.x_type_size = 4;
op_compile_info.x_align = 8;
op_compile_info.is_static = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"515000000", {10000, 20000, 30000, 40000, 50000}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[2] = {0, 0};
test.SetInt32ConstInput(1, begin_data, 2);
int32_t size_data[2] = {1, 20};
test.SetInt32ConstInput(2, size_data, 2);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "60, 0, 20, 20, 20";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_3) {
// compile info parse static
std::string compileInfo = R"({"_base_info":[32, 262144, 1],
"_coex_list": [3, 4, 2],
"_is_static": true,
"_const_sizes": [3, 4, 2],
"_const_begins": [0, 0, 0],
"_slice_vars": {"901210003": [10001, 10003, 20001, 30000, 40003]},
"_pattern":"Slice"})";

nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

SliceDslCompileInfo actual_struct("slice", op_info);
ASSERT_TRUE(actual_struct.is_static);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_4) {
// compile info parse dynamic static
std::string compileInfo = R"({"_base_info":[32, 262144, 2],
"_coex_list": [3, 4, 2],
"_const_sizes": [3, 4, 2],
"_const_begins": [0, 0, 0],
"_slice_vars": {"515000000": [10000, 20000, 30000, 40000, 50000]},
"_is_const": true,
"_const_info": [515000000, 32],
"_pattern":"Slice"})";

nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

SliceDslCompileInfo actual_struct("slice", op_info);
ASSERT_TRUE(actual_struct.is_const);
EXPECT_EQ(actual_struct.const_block_dims, 32);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_5
) {
// TODO const input condtion
// slice one dim large
vector<vector<int64_t>> inputs {
   {512, 768},
   {2},
   {2}
};
vector<vector<int64_t>> outputs{
   {256, 768},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 65280;
op_compile_info.x_type_size = 4;
op_compile_info.x_align = 8;
op_compile_info.is_static = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
   {"515000000", {10000, 20000, 30000, 40000, 50000}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[2] = {0, 0};
test.SetInt32ConstInput(1, begin_data, 2);
int32_t size_data[2] = {256, 768};
test.SetInt32ConstInput(2, size_data, 2);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "393216, 0, 196608, 24576, 12288";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_6
) {
// TODO const input condtion --- multi data error
// slice multi dims
vector<vector<int64_t>> inputs {
    {32, 35, 170, 4032},
    {4},
    {4}
};
vector<vector<int64_t>> outputs{
    {32, 11, 11, 672},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 65280;
op_compile_info.x_type_size = 4;
op_compile_info.x_align = 8;
op_compile_info.is_static = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"544000001", {10000, 10001, 10002, 10003, 20000, 30000, 20001, 30001, 20002, 30002, 20003, 30003, 40000, 50001}},
};
op_compile_info.is_valid = true;
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
//int32_t begin_data[2] = {0, 0};
//test.SetInt32ConstInput(1, begin_data, 2);
//int32_t size_data[2] = {32, 11};
//test.SetInt32ConstInput(2, size_data, 2);

int32_t begin_data[4] = {0, 0, 0, 0};
test.SetInt32ConstInput(1, begin_data, 4);
int32_t size_data[4] = {32, 11, 11, 672};
test.SetInt32ConstInput(2, size_data, 4);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "32, 35, 170, 4032, 0, 32, 0, 11, 0, 11, 0, 672, 1, 4";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_7
) {
// TODO const input condtion --- multi data error
// slice depad mode to small rows num change to data mov
vector<vector<int64_t>> inputs {
    {7, 8, 1, 3, 31},
    {5},
    {5}
};
vector<vector<int64_t>> outputs{
    {7, 8, 1, 3, 23},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 65280;
op_compile_info.x_type_size = 4;
op_compile_info.x_align = 8;
op_compile_info.is_static = false;
op_compile_info.end_mode = 0;
op_compile_info.is_const_ends = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"527000000", {10000, 10001, 20000, 30000, 20001, 30001, 40000, 50000}},
};
op_compile_info.is_valid = true;
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[5] = {0, 0, 0, 0, 5};
test.SetInt32ConstInput(1, begin_data, 5);
int32_t size_data[5] = {-1, -1, -1, -1, 23};
test.SetInt32ConstInput(2, size_data, 5);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "168, 31, 0, 168, 5, 23, 168, 168";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_8
) {
// TODO const input condtion --- multi data error
// slice depad b8
vector<vector<int64_t>> inputs {
    {7, 9029, 61},
    {3},
    {3}
};
vector<vector<int64_t>> outputs{
    {7, 9029, 2},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 261120;
op_compile_info.x_type_size = 1;
op_compile_info.x_align = 32;
op_compile_info.is_static = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"522000000", {10000, 10001, 20000, 30000, 20001, 30001, 40000, 50000}},
};
op_compile_info.is_valid = true;
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[3] = {0, 0, 36};
test.SetInt32ConstInput(1, begin_data, 3);
int32_t size_data[3] = {-1, -1, 2};
test.SetInt32ConstInput(2, size_data, 3);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "63203, 61, 0, 63203, 36, 2, 1976, 1976";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_9
) {
// TODO const input condtion --- multi data error
// slice depad b16
vector<vector<int64_t>> inputs {
    {1, 5, 7, 3},
    {4},
    {4}
};
vector<vector<int64_t>> outputs{
    {1, 5, 7, 1},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 130560;
op_compile_info.x_type_size = 2;
op_compile_info.x_align = 16;
op_compile_info.is_static = false;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"527000000", {10000, 10001, 20000, 30000, 20001, 30001, 40000, 50000}},
};
op_compile_info.is_valid = true;
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[4] = {0, 0, 0, 2};
test.SetInt32ConstInput(1, begin_data, 4);
int32_t size_data[4] = {-1, -1, -1, -1};
test.SetInt32ConstInput(2, size_data, 4);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "35, 3, 0, 35, 2, 1, 35, 35";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_10
) {
// TODO const input condtion --- multi data error
// slice depad b64
vector<vector<int64_t>> inputs {
    {16, 718141, 25, 2},
    {4},
    {4}
};
vector<vector<int64_t>> outputs{
    {16, 718141, 25, 1},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 32640;
op_compile_info.x_type_size = 8;
op_compile_info.x_align = 4;
op_compile_info.tensor_nums = 3;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"527000000", {10000, 10001, 20000, 30000, 20001, 30001, 40000, 50000}},
};
op_compile_info.is_valid = true;
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[4] = {0, 0, 0, 1};
test.SetInt32ConstInput(1, begin_data, 4);
int32_t size_data[4] = {-1, -1, -1, -1};
test.SetInt32ConstInput(2, size_data, 4);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "287256400, 2, 0, 287256400, 1, 1, 8976763, 2036";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}

TEST_F(SliceScheduleTilingRt2, slice_schedule_tiling_11
) {
// TODO const input condtion --- multi data error
// slice fuse slice 1 dims
vector<vector<int64_t>> inputs {
    {16, 718141, 25, 2},
    {4},
    {4}
};
vector<vector<int64_t>> outputs{
    {16, 718141, 1, 1},
};
ge::DataType dtype = ge::DT_INT32;

SliceDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::SLICE;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 32640;
op_compile_info.x_type_size = 8;
op_compile_info.x_align = 4;
op_compile_info.tensor_nums = 3;
op_compile_info.is_const_sizes = false;
op_compile_info.is_const_begins = false;
op_compile_info.end_mode = 0;
op_compile_info.coex_list = {2, 4, 3};

op_compile_info.slice_vars = {
    {"527000000", {10000, 10001, 20000, 30000, 20001, 30001, 40000, 50000}},
};
op_compile_info.is_valid = true;
AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t begin_data[4] = {0, 0, 0, 1};
test.SetInt32ConstInput(1, begin_data, 4);
int32_t size_data[4] = {-1, -1, 1, -1};
test.SetInt32ConstInput(2, size_data, 4);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "11490256, 50, 0, 11490256, 1, 1, 359071, 152";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}

