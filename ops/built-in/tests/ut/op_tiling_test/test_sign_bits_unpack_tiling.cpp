#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class SignBitsUnpackTiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "SignBitsUnpackTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SignBitsUnpackTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int64_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }
    return result;
}

TEST_F(SignBitsUnpackTiling, sign_bits_unpack_0) {
    using namespace optiling;
    std::string op_name = "SignBitsUnpack";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 248, \"align_unit\": 256, \"ub_spilt\": 4, \"num_each_block\": 8, \"core_num\": 32, \"dim\": 1}}";

    std::vector<int64_t> input{521};
    std::vector<int64_t> output{1, 521*8};

    TeOpTensor tensor_x;
    tensor_x.shape = input;
    tensor_x.dtype = "uint8";
    tensor_x.format = "ND";
    TeOpTensor tensor_y;
    tensor_y.shape = output;
    tensor_y.dtype = "float32";
    tensor_y.format = "ND";

    TeOpTensorArg tensor_input_arg;
    tensor_input_arg.tensor.push_back(tensor_x);
    tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_y);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "sign_bits_unpack_0";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "17 1 1 0 1 0 1 184 62 522 4168 ");
}
