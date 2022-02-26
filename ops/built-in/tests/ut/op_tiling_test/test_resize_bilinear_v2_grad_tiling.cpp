#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class ResizeBilinearV2GradTiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "ResizeBilinearV2GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ResizeBilinearV2GradTiling TearDown" << std::endl;
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

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_10) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

    std::vector<int64_t> input0{2, 1, 3200, 3000, 16};
    std::vector<int64_t> input1{2, 1, 120, 160, 16};
    std::vector<int64_t> output{2, 1, 120, 160, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456787";
    OpRunInfo runInfo;
    ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_0) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{32, 32, 32, 32, 16};
    std::vector<int64_t> input1{32, 32, 16, 16, 16};
    std::vector<int64_t> output{32, 32, 16, 16, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);\
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 32 32 32 0 0 32 32 16 16 16384 4096 512 256 1024 1 32 ");
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_1) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 28, 56, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);\
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345673";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 2 1 1 0 0 28 56 7 8 25088 896 896 128 2 1 56 ");
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_2) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 28, 56, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);\
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345672";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_3) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 7, 8, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456783";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_4) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 7, 8, 16};
    std::vector<int64_t> input1{2, 1, 1, 1, 16};
    std::vector<int64_t> output{2, 1, 1, 1, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456784";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_5) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 64, 64, 16};
    std::vector<int64_t> input1{2, 1, 1, 1, 16};
    std::vector<int64_t> output{2, 1, 1, 1, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456785";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_6) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 1, 1, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456786";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_7) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 3200, 3000, 16};
    std::vector<int64_t> input1{2, 1, 120, 160, 16};
    std::vector<int64_t> output{2, 1, 120, 160, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456787";
    OpRunInfo runInfo;
    ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_8) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 2, 96, 128, 16};
    std::vector<int64_t> input1{2, 2, 48, 64, 16};
    std::vector<int64_t> output{2, 2, 48, 64, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);\
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "5 32 0 0 3 3 96 128 48 64 196608 49152 2048 1024 4 1 128 ");
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_9) {
    using namespace optiling;
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1}}";

    std::vector<int64_t> input0{2, 1, 3200, 3000, 16};
    std::vector<int64_t> input1{2, 1, 120, 160, 16};
    std::vector<int64_t> output{2, 1, 120, 160, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float32";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float32";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "123456787";
    OpRunInfo runInfo;
    ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
