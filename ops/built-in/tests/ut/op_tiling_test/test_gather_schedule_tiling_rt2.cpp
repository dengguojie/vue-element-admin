#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/gather_dsl.h"

#include "common_autotiling_util.h"
#include "graph/utils/attr_utils.h"

#define private public
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

using namespace optiling;
class GatherScheduleTilingRt2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherScheduleTilingRt2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherScheduleTilingRt2 TearDown" << std::endl;
  }
};

TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_0
) {
// gather static
vector<vector<int64_t>> inputs {
    {1, 1, 1, 100},
    {1, 120879}
};
vector<vector<int64_t>> outputs{
    {1, 1, 120879, 100},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 2;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"0", {116480, 7280}},
};

op_compile_info.is_dynamic_const = true;
op_compile_info.const_axis = 2;
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "900010010, 2, 3778, 2, 756";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_1
) {
// gather nd static
vector<vector<int64_t>> inputs {
    {1, 24, 512, 128, 1},
    {1, 25165824, 3}
};
vector<vector<int64_t>> outputs{
    {1, 25165824, 1},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 1;
op_compile_info.params_dtype = 2;
op_compile_info.params_align = 16;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"0", {116480, 7280}},
};

op_compile_info.const_axis = 0;
op_compile_info.is_dynamic_const = true;
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "900030004, 1, 786432, 1, 2420";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_2
) {
// gather v2 static
vector<vector<int64_t>> inputs {
    {1, 1, 1, 1},
    {1, 1},
    {1}
};
vector<vector<int64_t>> outputs{
    {1, 1, 1, 1},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 2;
op_compile_info.params_align = 16;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"0", {116480, 7280}},
};

op_compile_info.const_axis = 0;
op_compile_info.is_dynamic_const = true;
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
//int32_t index_data[1] = {0};
//test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "900010000, 0, 1, 0, 1";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_3
) {
// gather v2 db
vector<vector<int64_t>> inputs {
    {500, 400, 16, 1},
    {1},
    {1}
};
vector<vector<int64_t>> outputs{
    {1, 400, 16, 1},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"0", {54280, 3640}},
};

op_compile_info.gather_vars = {
    {"900010015", {10001, 10002, 10003, 30003, 40003}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {0};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "1, 500, 6400, 200, 200";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_4
) {
// gather v2 remov pad
vector<vector<int64_t>> inputs {
    {1901},
    {3120},
    {1}
};
vector<vector<int64_t>> outputs{
    {3120},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 49152;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"6", {29120, 3640}},
    {"0", {29120, 3640}},
};

op_compile_info.gather_vars = {
    {"900016010", {10001, 10002, 10003, 20001, 30002, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {0};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "1, 1901, 1, 3120, 98, 98";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_5
) {
// gather v2 scalar mode
vector<vector<int64_t>> inputs {
    {162},
    {1784},
    {1}
};
vector<vector<int64_t>> outputs{
    {1784},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"6", {29120, 3640}},
};

op_compile_info.gather_vars = {
    {"900016010", {10001, 10002, 10003, 20001, 30002, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {0};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "1, 162, 1, 1784, 56, 56";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_6
) {
// gather v2 first axis
vector<vector<int64_t>> inputs {
    {5, 3, 3, 16, 2, 32},
    {5, 11, 7},
    {1}
};
vector<vector<int64_t>> outputs{
    {5, 3, 11, 7, 16, 2, 32},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 1;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 1;

op_compile_info.tensor_sizes = {
    {"0", {58240, 7280}},
};

op_compile_info.gather_vars = {
    {"900010010", {10001, 10002, 10003, 20001, 30000, 40003}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {2};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "3, 3, 1024, 77, 3, 3";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_7
) {
// gather v2 second axis
vector<vector<int64_t>> inputs {
    {7, 1, 16, 3, 16},
    {3, 158, 5, 3},
    {1}
};
vector<vector<int64_t>> outputs{
    {7, 3, 158, 5, 3, 16, 3, 16},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 131072;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"2", {116480, 3640}},
    {"0", {116480, 3640}},
};

op_compile_info.gather_vars = {
    {"900011010", {10001, 10002, 10003, 20001, 30001, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {1};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "7, 1, 768, 7110, 223, 112";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_8
) {
// _gather_vars exception
vector<vector<int64_t>> inputs {
    {7, 1, 16, 3, 16},
    {3, 158, 5, 3},
    {1}
};
vector<vector<int64_t>> outputs{
    {7, 3, 158, 5, 3, 16, 3, 16},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 131072;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"2", {116480, 3640}},
    {"0", {116480, 3640}},
};

op_compile_info.gather_vars = {
    {"600011010", {10001, 10002, 10003, 20001, 30001, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {1};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), false);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_9
) {
// gather v2 block first axis, ub first axis
vector<vector<int64_t>> inputs {
    {40, 1, 5},
    {40, 8, 17, 16},
    {1}
};
vector<vector<int64_t>> outputs{
    {40, 1, 8, 17, 16},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 1;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 1;

op_compile_info.tensor_sizes = {
    {"0", {58240, 7280}},
};

op_compile_info.gather_vars = {
    {"900010000", {10000, 10001, 10002, 10003, 20001, 30000, 40000}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {-1};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "40, 1, 5, 1, 2176, 2, 2";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}

TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_11
) {
// gather v2 remov pad uint8
vector<vector<int64_t>> inputs {
    {261, 16},
    {61},
    {1}
};
vector<vector<int64_t>> outputs{
    {61, 16},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 131072;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"7", {63488, 1984}},
};

op_compile_info.gather_vars = {
    {"900017000", {10001, 10002, 10003, 20001, 30002, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {0};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "1, 261, 16, 61, 1, 1";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_13
) {
// gather v2 remov pad float16
vector<vector<int64_t>> inputs {
    {2, 7, 2, 2, 7, 3},
    {3},
    {1}
};
vector<vector<int64_t>> outputs{
    {2, 7, 3, 2, 7, 3},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 2;
op_compile_info.params_align = 16;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"7", {31744, 1984}},
};

op_compile_info.gather_vars = {
    {"900017005", {10001, 10002, 10003, 20001, 30002, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {2};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "14, 2, 42, 3, 1, 1";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_14
) {
// gather nd broadcast
vector<vector<int64_t>> inputs {
    {1, 24, 512, 128, 1},
    {1, 25165824, 0},
};
vector<vector<int64_t>> outputs{
    {1, 25165824, 3},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 1;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 1;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"9", {0, 0}},
};

op_compile_info.gather_vars = {
    {"990000001", {10001, 2001}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
EXPECT_EQ(test.Test(), true);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_15
) {
// gather v2 shape 0
vector<vector<int64_t>> inputs {
    {1, 0},
    {3120},
    {1}
};
vector<vector<int64_t>> outputs{
    {3120},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 8;
op_compile_info.params_align = 4;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 16384;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"7", {7936, 1984}},
};

op_compile_info.gather_vars = {
    {"990000000", {10001, 10002, 10003, 20001, 30002, 40002}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
int32_t index_data[1] = {0};
test.SetInt32ConstInput(2, index_data, 1);
EXPECT_EQ(test.Test(), true);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_16
) {
// gather v2 ub align static
vector<vector<int64_t>> inputs {
    {1, 1, 26, 200},
    {1, 15360},
    {1}
};
vector<vector<int64_t>> outputs{
    {3120},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;
op_compile_info.const_axis = 2;
op_compile_info.is_dynamic_const = true;
op_compile_info.is_valid = true;

op_compile_info.tensor_sizes = {
    {"1", {29120, 3640}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "900011010, 2, 480, 2, 120";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_17
) {
// gather v2 ub align static
vector<vector<int64_t>> inputs {
    {1, 1, 26, 100},
    {1, 15360},
    {1}
};
vector<vector<int64_t>> outputs{
    {3120},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = false;
op_compile_info.org_batch_dims = 0;
op_compile_info.const_axis = 2;
op_compile_info.is_dynamic_const = true;

op_compile_info.tensor_sizes = {
    {"2", {29120, 3640}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "900012010, 2, 480, 2, 240";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}


TEST_F(GatherScheduleTilingRt2, gather_schedule_tiling_18
) {
// gather v2 binary condtion
vector<vector<int64_t>> inputs {
    {1, 1, 26, 100},
    {1, 15360},
    {1}
};
vector<vector<int64_t>> outputs{
    {3120},
};
ge::DataType dtype = ge::DT_INT32;

GatherDslCompileInfo op_compile_info;
op_compile_info.pattern = SchPattern::GATHER;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.unknown_batch_dims = true;
op_compile_info.org_batch_dims = 0;
op_compile_info.const_axis = 2;
op_compile_info.is_dynamic_const = true;
op_compile_info.attr_name = "batch_dims";


op_compile_info.tensor_sizes = {
    {"2", {29120, 3640}},
};
op_compile_info.is_valid = true;

AutoTilingTest test(inputs, outputs, dtype, dtype);
test.SetCompileInfo(&op_compile_info);
std::vector<std::pair<string, int64_t>> gather_attr = {{"batch_dims", -2}};
test.SetAttrs<int64_t>(gather_attr);
EXPECT_EQ(test.Test(), true);
std::string expect_tiling_data = "900012010, 2, 480, 2, 240";
EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
}
