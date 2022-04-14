#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"

#define private public

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
#include "op_tiling/slice_dsl.h"
#include "op_tiling/tiling_handler.h"

using namespace std;
using namespace ge;


class SliceScheduleTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SliceScheduleTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SliceScheduleTiling TearDown" << std::endl;
  }
};

/*
 * be careful of the to_string fuction
 * the type of tiling_data in other ops is int64 while int32 here
 */
static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy_s(&tmp, sizeof(&tmp), data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

TEST_F(SliceScheduleTiling, slice_schedule_tiling_0
) {
// slice static lr depad
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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

std::vector <int64_t> inputA{
    32509728, 25
};
std::vector <int64_t> output{32509728, 20};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::SliceD("SliceD");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "527000000 0 1015929 0 504 7 32 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_1
) {
// slice data mov
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{512, 50, 1324};
std::vector <int64_t> inputB{3};
std::vector <int64_t> inputC{3};
std::vector <int64_t> output{32509728, 20};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 0};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{-1, -1, 512};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "25600 1324 0 25600 0 512 800 112 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_2
) {
// slice one dim
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{3, 20};
std::vector <int64_t> inputB{2};
std::vector <int64_t> inputC{2};
std::vector <int64_t> output{1, 20};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{1, 20};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "60 0 20 20 20 ");
}

TEST_F(SliceScheduleTiling, slice_schedule_tiling_3) {
// compile info parse static
using namespace optiling;
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


TEST_F(SliceScheduleTiling, slice_schedule_tiling_4) {
// compile info parse dynamic static
using namespace optiling;
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

TEST_F(SliceScheduleTiling, slice_schedule_tiling_5
) {
// slice one dim large
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{512, 768};
std::vector <int64_t> inputB{2};
std::vector <int64_t> inputC{2};
std::vector <int64_t> output{256, 768};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{256, 768};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "393216 0 196608 24576 12288 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_6
) {
// slice multi dims
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{32, 35, 170, 4032};
std::vector <int64_t> inputB{4};
std::vector <int64_t> inputC{4};
std::vector <int64_t> output{32, 11, 11, 672};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 0, 0};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{32, 11, 11, 672};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 35 170 4032 0 32 0 11 0 11 0 672 1 4 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_7
) {
// slice depad mode to small rows num change to data mov
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{7, 8, 1, 3, 31};
std::vector <int64_t> inputB{5};
std::vector <int64_t> inputC{5};
std::vector <int64_t> output{7, 8, 1, 3, 23};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT16);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 0, 0, 5};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{-1, -1, -1, -1, 23};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "168 31 0 168 5 23 168 168 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_8
) {
// slice depad b8
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{7, 9029, 61};
std::vector <int64_t> inputB{3};
std::vector <int64_t> inputC{3};
std::vector <int64_t> output{7, 9029, 2};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT8);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 36};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{-1, -1, 2};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "63203 61 0 63203 36 2 1976 1976 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_9
) {
// slice depad b16
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{1, 5, 7, 3};
std::vector <int64_t> inputB{4};
std::vector <int64_t> inputC{4};
std::vector <int64_t> output{1, 5, 7, 1};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT16);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 0, 2};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{-1, -1, -1, -1};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "35 3 0 35 2 1 35 35 ");
}


TEST_F(SliceScheduleTiling, slice_schedule_tiling_10
) {
// slice depad b64
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{16, 718141, 25, 2};
std::vector <int64_t> inputB{4};
std::vector <int64_t> inputC{4};
std::vector <int64_t> output{16, 718141, 25, 1};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT16);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 0, 1};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{-1, -1, -1, -1};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "287256400 2 0 287256400 1 1 8976763 2036 ");
}

TEST_F(SliceScheduleTiling, slice_schedule_tiling_11
) {
// slice fuse slice 1 dims
using namespace optiling;
SliceDslCompileInfo op_compile_info;
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


std::vector <int64_t> inputA{16, 718141, 25, 2};
std::vector <int64_t> inputB{4};
std::vector <int64_t> inputC{4};
std::vector <int64_t> output{16, 718141, 1, 1};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_FLOAT16);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
vector<int32_t> begin_data{0, 0, 0, 1};

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> size_data{-1, -1, 1, -1};

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::Slice("Slice");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT_CONST(opParas, tensor_inputB, offsets, (const uint8_t*)begin_data.data(), begin_data.size() * 4);
TENSOR_INPUT_CONST(opParas, tensor_inputC, size, (const uint8_t*)size_data.data(), size_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::SliceDsl slice_dsl_schedule("slice", opParas, op_compile_info, runInfo);
slice_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "11490256 50 0 11490256 1 1 359071 152 ");
}
