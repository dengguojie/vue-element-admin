#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include <iostream>
using namespace std;

class UnsortedSegmentMaxTiling : public testing::Test {
 protected:
    static void SetUpTestCase() {
        std::cout << "UnsortedSegmentMaxTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "UnsortedSegmentMaxTiling TearDown" << std::endl;
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

TEST_F(UnsortedSegmentMaxTiling, unsortedsegmentmax_tiling_0) {
    using namespace optiling;
    std::string op_name = "UnsortedSegmentMax";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentMax");
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    
    std::string compileInfo = "{\"vars\": {\"ub_size\": 261632, \"core_num\": 32, \"dtype\":\"float16\", \"ub_tensor_num\":2}}";

    std::vector<int64_t> inputA{3, 16, 10419, 3};
    std::vector<int64_t> inputB{3};
    std::vector<int64_t> inputC{1};
    std::vector<int32_t> num_segments{8,};
    std::vector<int64_t> output{8, 16, 10419, 3};

    TeOpTensor tensor_inputA;
    tensor_inputA.shape = inputA;
    tensor_inputA.dtype = "float16";
    TeOpTensor tensor_inputB;
    tensor_inputB.shape = inputB;
    tensor_inputB.dtype = "int32";
    TeOpTensor tensor_inputC;
    tensor_inputC.shape = inputC;
    tensor_inputC.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(tensor_inputA);
    tensor_argA.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argB;
    tensor_argB.tensor.push_back(tensor_inputB);
    tensor_argB.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argC;
    tensor_argC.tensor.push_back(tensor_inputC);
    tensor_argC.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.inputs.push_back(tensor_argB);
    opParas.inputs.push_back(tensor_argC);
    opParas.outputs.push_back(tensor_arg);
    opParas.op_type = op_name;
    opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "50 32 3 3 1 3 3 96 5248 1552 1 8 8 1 1 1 5248 500112 41 13 0 328 97 1 ");
}
TEST_F(UnsortedSegmentMaxTiling, unsortedsegmentmax_tiling_1) {
    using namespace optiling;
    std::string op_name = "UnsortedSegmentMax";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentMax");
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    
    std::string compileInfo = "{\"vars\": {\"ub_size\": 261632, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

    std::vector<int64_t> inputA{4, 31, 4};
    std::vector<int64_t> inputB{4};
    std::vector<int64_t> inputC{1};
    std::vector<int32_t> num_segments{16,};
    std::vector<int64_t> output{16, 31, 4};

    TeOpTensor tensor_inputA;
    tensor_inputA.shape = inputA;
    tensor_inputA.dtype = "float32";
    TeOpTensor tensor_inputB;
    tensor_inputB.shape = inputB;
    tensor_inputB.dtype = "int32";
    TeOpTensor tensor_inputC;
    tensor_inputC.shape = inputC;
    tensor_inputC.dtype = "int32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(tensor_inputA);
    tensor_argA.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argB;
    tensor_argB.tensor.push_back(tensor_inputB);
    tensor_argB.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argC;
    tensor_argC.tensor.push_back(tensor_inputC);
    tensor_argC.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_output);
    tensor_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.inputs.push_back(tensor_argB);
    opParas.inputs.push_back(tensor_argC);
    opParas.outputs.push_back(tensor_arg);
    opParas.op_type = op_name;
    opParas.const_inputs["num_segments"] =
            std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)num_segments.data(), num_segments.size() * 4, ge::Tensor());
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "40 2 1 1 1 4 4 2 64 60 8 2 2 8 1 1 64 124 1 1 4 8 8 1 ");
}
