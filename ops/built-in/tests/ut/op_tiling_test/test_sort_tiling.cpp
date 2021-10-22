#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class SortTiling : public testing::Test {
 protected:
    static void SetUpTestCase() {
        std::cout << "SortTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SortTiling TearDown" << std::endl;
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

TEST_F(SortTiling, sort_tiling_0) {
    using namespace optiling;
    std::string op_name = "Sort";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Sort");
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    
    std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"core_num\": 32}}";

    std::vector<int64_t> input_tensor_shape{2, 2, 2, 5600};

    std::vector<int64_t> out_tensor_shape{2, 2, 2, 5600};
    std::vector<int64_t> out_indices_tensor_shape{2, 2, 2, 5600};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensor out_tensor;
    out_tensor.shape = out_tensor_shape;
    out_tensor.dtype = "float16";

    TeOpTensor out_indices_tensor;
    out_indices_tensor.shape = out_indices_tensor_shape;
    out_indices_tensor.dtype = "int32";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpTensorArg tensor_argC;
    tensor_argC.tensor.push_back(out_tensor);
    tensor_argC.arg_type = TA_SINGLE;

    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(out_indices_tensor);
    tensor_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.outputs.push_back(tensor_argC);
    opParas.outputs.push_back(tensor_arg);

    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "babababaing";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 8 8 5600 5600 6 0 8 5 6144 ");
}

