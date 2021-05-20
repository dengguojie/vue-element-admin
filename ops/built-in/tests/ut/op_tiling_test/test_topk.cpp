#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include <iostream>
using namespace std;

class TopKDTiling : public testing::Test {
 protected:
    static void SetUpTestCase() {
        std::cout << "TopKDTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "TopKDTiling TearDown" << std::endl;
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

TEST_F(TopKDTiling, topkd_tiling_0) {
    using namespace optiling;
    std::string op_name = "TopKD";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("TopKD");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    
    std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"batch_cols_padding\":4835, \"k_num\":139, \"max_k\":4096}}";

    std::vector<int64_t> input_tensor_shape{32, 20308};
    std::vector<int64_t> indices_tensor_shape{8192};
    std::vector<int64_t> out_tensor_shape{139};
    std::vector<int64_t> out_indices_tensor_shape{139};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";
    TeOpTensor indices_tensor;
    indices_tensor.shape = indices_tensor_shape;
    indices_tensor.dtype = "int32";
    TeOpTensor out_tensor;
    out_tensor.shape = out_tensor_shape;
    out_tensor.dtype = "float16";
    TeOpTensor out_indices_tensor;
    out_indices_tensor.shape = out_indices_tensor_shape;
    out_indices_tensor.dtype = "int32";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argB;
    tensor_argB.tensor.push_back(indices_tensor);
    tensor_argB.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_argC;
    tensor_argC.tensor.push_back(out_tensor);
    tensor_argC.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(out_indices_tensor);
    tensor_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.inputs.push_back(tensor_argB);
    opParas.inputs.push_back(tensor_argC);
    opParas.outputs.push_back(tensor_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "32 32 20308 139 6 1 1 32 ");
}

