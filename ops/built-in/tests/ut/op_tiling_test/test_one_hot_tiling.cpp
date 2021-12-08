#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class OneHotTiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "OneHotTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "OneHotTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_F(OneHotTiling, one_hot_tiling_0) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

    std::vector<int64_t> input0{3, 3, 32, 32, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{3, 3, 32, 32, 16, 32};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 4608 2 32 147456 147456 1 4718592 4608 0 0 ");
}

TEST_F(OneHotTiling, one_hot_tiling_1) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":0}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{32, 2, 16, 8, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_2) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":0}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{32, 2, 16, 8, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_3) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 16, 8, 8, 16, 32};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_4) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 32, 16, 8, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_5) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":2}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 16, 32, 8, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_6) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":3}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 16, 8, 32, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_7) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":4}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 16, 8, 8, 32, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_8) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 2, 8, 8, 16, 8, 32};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_9) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":0}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{32, 2, 2, 8, 8, 16, 8};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_10) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 32, 2, 8, 8, 16, 8};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_11) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":2}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 2, 32, 8, 8, 16, 8};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_12) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":3}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 2, 8, 32, 8, 16, 8};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_13) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\": 4}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 2, 8, 8, 32, 16, 8};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_14) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":5}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 2, 8, 8, 16, 32, 8};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_15) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":-1}}";

    std::vector<int64_t> input0{2, 2, 8, 8, 16, 8, 2};
    std::vector<int32_t> depth{16};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 2, 8, 8, 16, 8, 2, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 9 16 65536 1 65536 1048576 0 1 1 ");
}

TEST_F(OneHotTiling, one_hot_tiling_16) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

    std::vector<int64_t> input0{2, 21120};
    std::vector<int32_t> depth{323};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 323, 21120};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 9 30 42240 1 42240 13643520 0 11 4 ");
}

TEST_F(OneHotTiling, one_hot_tiling_17) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":1}}";

    std::vector<int64_t> input0{1, 4224};
    std::vector<int32_t> depth{323};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{1, 323, 4224};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 7 30 4224 1 4224 1364352 0 11 4 ");
}
TEST_F(OneHotTiling, one_hot_tiling_18) {
    using namespace optiling;
    std::string op_name = "OneHot";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"axis\":3}}";

    std::vector<int64_t> input0{2, 16, 8, 8, 16};
    std::vector<int32_t> depth{32};
    std::vector<int32_t> off_value{1};
    std::vector<int64_t> output{2, 32, 16, 8, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "int32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.const_inputs["depth"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)depth.data(), depth.size() * 4, ge::Tensor());
    opParas.const_inputs["off_value"] = std::tuple<const uint8_t*, size_t, ge::Tensor>(
    (const uint8_t*)off_value.data(), off_value.size() * 4, ge::Tensor());
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 0 8 32 32768 1 32768 1048576 0 1 1 ");
}