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
#include "op_tiling/gather_dsl.h"
#include "op_tiling/tiling_handler.h"

using namespace std;
using namespace ge;


class GatherScheduleTiling : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "GatherScheduleTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherScheduleTiling TearDown" << std::endl;
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

TEST_F(GatherScheduleTiling, gather_schedule_tiling_0
) {
// gather static
using namespace optiling;
GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 2;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"0", {116480, 7280}},
};

op_compile_info.is_dynamic_const = true;
op_compile_info.const_axis = 2;

std::vector <int64_t> inputA{
        1, 1, 1, 100
};
std::vector <int64_t> inputB{1, 120879};
std::vector <int64_t> output{1, 1, 120879, 100};

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
SetOriginShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);
TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT16);

auto opParas = op::Gather("Gather");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "900210010 2 3778 2 756 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_1
) {
// gather nd static
using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 1;
op_compile_info.params_dtype = 2;
op_compile_info.params_align = 16;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
    {"0", {116480, 7280}},
};

op_compile_info.const_axis = 0;
op_compile_info.is_dynamic_const = true;

std::vector <int64_t> inputA{
        1, 24, 512, 128, 1
};
std::vector <int64_t> inputB{1, 25165824, 3};
std::vector <int64_t> output{1, 25165824, 1};

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
SetOriginShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT16);

auto opParas = op::GatherNd("GatherNd");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "900030004 1 786432 1 2420 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_2
) {
// gather v2 static
using namespace optiling;
GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 2;
op_compile_info.params_align = 16;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"0", {116480, 7280}},
};

op_compile_info.const_axis = 0;
op_compile_info.is_dynamic_const = true;

std::vector <int64_t> inputA{
        1, 1, 1, 1
};
std::vector <int64_t> inputB{1, 1};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{1, 1, 1, 1};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT16);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();

EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
"900210000 0 1 0 1 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_3
) {
// gather v2 db
using namespace optiling;
GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"0", {54280, 3640}},
};

op_compile_info.gather_vars = {
        {"900210015", {10001, 10002, 10003, 30003, 40003}},
};

std::vector <int64_t> inputA{
        500, 400, 16, 1
};
std::vector <int64_t> inputB{1};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{1, 400, 16, 1};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT16);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
"1 500 6400 200 200 ");
}


TEST_F(GatherScheduleTiling, gather_schedule_tiling_4
) {
// gather v2 remov pad

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 49152;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"6", {29120, 3640}},
        {"0", {29120, 3640}},
};

op_compile_info.gather_vars = {
        {"900216010", {10001, 10002, 10003, 20001, 30002, 40002}},
};

std::vector <int64_t> inputA{1901};
std::vector <int64_t> inputB{3120};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{3120};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1901 1 3120 98 98 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_5
) {
// gather v2 scalar mode

using namespace optiling;
GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"6", {29120, 3640}},
};

op_compile_info.gather_vars = {
        {"900216010", {10001, 10002, 10003, 20001, 30002, 40002}},
};

std::vector <int64_t> inputA{162};
std::vector <int64_t> inputB{1784};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{1784};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();

EXPECT_EQ(to_string(runInfo.GetAllTilingData()),"1 162 1 1784 56 56 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_6
) {
// gather v2 first axis

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 1;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 1;

op_compile_info.tensor_sizes = {
        {"0", {58240, 7280}},
};

op_compile_info.gather_vars = {
        {"901210002", {10001, 10002, 10003, 20001, 30000, 40003}},
};

std::vector <int64_t> inputA{5, 3, 3, 16, 2, 32};
std::vector <int64_t> inputB{5, 11, 7};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{5, 3, 11, 7, 16, 2, 32};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{2};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();

EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
"");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_7
) {
// gather v2 second axis

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 131072;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"2", {116480, 3640}},
        {"0", {116480, 3640}},
};

op_compile_info.gather_vars = {
        {"900211010", {10001, 10002, 10003, 20001, 30001, 40002}},
};

std::vector <int64_t> inputA{7, 1, 16, 3, 16};
std::vector <int64_t> inputB{3, 158, 5, 3};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{7, 3, 158, 5, 3, 16, 3, 16};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{1};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();

EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
"7 1 768 7110 223 112 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_8
) {
// _gather_vars exception

using namespace optiling;
std::string compileInfo = R"({"_base_info":[32, 262144, 0, 4, 4],
"_custom_info":[58240, 0, false, 0],
"_vars": {"901210002": ["_params_dims_0", "_params_dims_1", "_params_dims_2", "_indices_dims_1",
"_block_factor_0", "_ub_factor_0"]}, "_pattern":"Gather"})";

std::vector <int64_t> inputA{40, 1, 5};
std::vector <int64_t> inputB{40, 8, 17, 16};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{40, 1, 8, 17, 16};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{-1};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
const nlohmann::json &parsed_compile_info = nlohmann::json::parse(compileInfo);
std::shared_ptr <AutoTilingHandler> outer_compile_info = \
    CreateGatherTilingHandler(this->test_info_->name(),
                              "Gather",
                              nlohmann::json::parse(compileInfo));
ASSERT_FALSE(outer_compile_info
->
DoTiling(opParas, runInfo
));
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_9
) {
// gather v2 block first axis, ub first axis

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 1;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 1;

op_compile_info.tensor_sizes = {
        {"0", {58240, 7280}},
};

op_compile_info.gather_vars = {
        {"901210000", {10000, 10001, 10002, 10003, 20001, 30000, 40000}},
};

std::vector <int64_t> inputA{40, 1, 5};
std::vector <int64_t> inputB{40, 8, 17, 16};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{40, 1, 8, 17, 16};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{-1};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();

EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
"40 1 5 1 2176 2 2 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_10
) {
// _custom_info len exception

using namespace optiling;
std::string compileInfo = R"({"_base_info":[32, 262144, 0, 4, 4],
"_custom_info":[58240, 7280, 262144, 32786, 1, 1, 8, false],
"_gather_vars": {"901210000": [10000, 10001, 10002, 20001, 30000, 40000]},
"_vars": {"901210002": ["_params_dims_0", "_params_dims_1", "_params_dims_2", "_indices_dims_1",
"_block_factor_0", "_ub_factor_0"]}, "_pattern":"Gather"})";

std::vector <int64_t> inputA{40, 1, 5};
std::vector <int64_t> inputB{40, 8, 17, 16};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{40, 1, 8, 17, 16};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{-1};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
const nlohmann::json &parsed_compile_info = nlohmann::json::parse(compileInfo);
std::shared_ptr <AutoTilingHandler> outer_compile_info = \
    CreateGatherTilingHandler(this->test_info_->name(),
                              "Gather",
                              nlohmann::json::parse(compileInfo));
ASSERT_FALSE(outer_compile_info
->
DoTiling(opParas, runInfo
));
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_11
) {
// gather v2 remov pad uint8

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 131072;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"7", {63488, 1984}},
};

op_compile_info.gather_vars = {
        {"900217000", {10001, 10002, 10003, 20001, 30002, 40002}},
};

std::vector <int64_t> inputA{261, 16};
std::vector <int64_t> inputB{61};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{61, 16};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 261 16 61 1 1 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_12
) {
// gather v2 remov pad uint8

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 131072;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"7", {63488, 1984}},
};

op_compile_info.gather_vars = {
        {"900217000", {10001, 10002, 10003, 20001, 30002, 40002}},
};

std::vector <int64_t> inputA{261, 16};
std::vector <int64_t> inputB{61};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{61, 16};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 261 16 61 1 1 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_13
) {
// gather v2 remov pad float16

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 2;
op_compile_info.params_align = 16;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"7", {31744, 1984}},
};

op_compile_info.gather_vars = {
        {"900217005", {10001, 10002, 10003, 20001, 30002, 40002}},
};

std::vector <int64_t> inputA{2, 7, 2, 2, 7, 3};
std::vector <int64_t> inputB{3
};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{2, 7, 3, 2, 7, 3};

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

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{2};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "14 2 42 3 1 1 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_14
) {
// gather nd broadcast
using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 1;
op_compile_info.params_dtype = 1;
op_compile_info.params_align = 32;
op_compile_info.indices_dtype = 1;

op_compile_info.params_ub_store_num = 65536;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"9", {0, 0}},
};

op_compile_info.gather_vars = {
        {"990000001", {10001, 2001}},
};

std::vector <int64_t> inputA{
        1, 24, 512, 128, 1
};
std::vector <int64_t> inputB{1, 25165824, 0};
std::vector <int64_t> output{1, 25165824, 3};

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
SetOriginShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_FLOAT16);

auto opParas = op::GatherNd("GatherNd");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(runInfo.GetBlockDim(), 1);
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_15
) {
// gather v2 remov pad uint64

using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 8;
op_compile_info.params_align = 4;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 16384;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;

op_compile_info.tensor_sizes = {
        {"7", {7936, 1984}},
};

op_compile_info.gather_vars = {
        {"900217010", {10001, 10002, 10003, 20001, 30002, 40002}},
};

std::vector <int64_t> inputA{1, 0};
std::vector <int64_t> inputB{3120};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{3120};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(runInfo.GetBlockDim(), 1);
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_16
) {
// gather v2 ub align static
using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;
op_compile_info.const_axis = 2;
op_compile_info.is_dynamic_const = true;


op_compile_info.tensor_sizes = {
        {"1", {29120, 3640}},
};

std::vector <int64_t> inputA{1, 1, 26, 200};
std::vector <int64_t> inputB{1, 15360};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{3120};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "900211010 2 480 2 120 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_17
) {
// gather v2 ub align static
using namespace optiling;

GatherDslCompileInfo op_compile_info;
op_compile_info.core_num = 32;
op_compile_info.ub_size = 262144;
op_compile_info.gather_type = 0;
op_compile_info.params_dtype = 4;
op_compile_info.params_align = 8;
op_compile_info.indices_dtype = 4;

op_compile_info.params_ub_store_num = 32768;
op_compile_info.batch_dims = 0;
op_compile_info.is_binary_shape = false;
op_compile_info.org_batch_dims = 0;
op_compile_info.const_axis = 2;
op_compile_info.is_dynamic_const = true;


op_compile_info.tensor_sizes = {
        {"2", {29120, 3640}},
};

std::vector <int64_t> inputA{1, 1, 26, 100};
std::vector <int64_t> inputB{1, 15360};
std::vector <int64_t> inputC{1};
std::vector <int64_t> output{3120};

TensorDesc tensor_inputA;
tensor_inputA.
SetShape(ge::Shape(inputA)
);
tensor_inputA.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputB;
tensor_inputB.
SetShape(ge::Shape(inputB)
);
tensor_inputB.
SetDataType(ge::DT_INT32);

TensorDesc tensor_inputC;
tensor_inputC.
SetShape(ge::Shape(inputC)
);
tensor_inputC.
SetDataType(ge::DT_INT32);
vector<int32_t> axis_data{0};


TensorDesc tensor_output;
tensor_output.
SetShape(ge::Shape(output)
);
tensor_output.
SetDataType(ge::DT_INT32);

auto opParas = op::GatherV2("GatherV2");
TENSOR_INPUT(opParas, tensor_inputA, x);
TENSOR_INPUT(opParas, tensor_inputB, indices);
TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis_data.data(), axis_data.size() * 4);
TENSOR_OUTPUT(opParas, tensor_output, y);

optiling::utils::OpRunInfo runInfo;
optiling::GatherDsl gather_dsl_schedule("gather", opParas, op_compile_info, runInfo);
gather_dsl_schedule.DoTiling();
EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "900212010 2 480 2 240 ");
}

TEST_F(GatherScheduleTiling, gather_schedule_tiling_19) {
using namespace optiling;
std::string compileInfo = R"({"_base_info":[32, 262144, 1048576, 0, 4, 4],
"_custom_info":[262144, 32786, 0, false, 0],
"_tensor_sizes": {"901210003": [10001, 10003, 20001, 30000, 40003]},
"_gather_vars": {"0": [209964, 6552]},
"_vars": {"901210002": ["_params_dims_1", "_params_dims_3", "_indices_dims_1", "_block_factor_0", "_ub_factor_3"]}, "_pattern":"Gather"})";

nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

GatherDslCompileInfo actual_struct("gather", op_info);
ASSERT_TRUE(actual_struct.org_batch_dims == 0);
}
