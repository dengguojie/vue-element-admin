#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include <register/op_tiling.h>
#include "array_ops.h"
#include "image_ops.h"
#include "test_common.h"

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

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_10) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";

    std::vector<int64_t> input0{2, 1, 3200, 3000, 16};
    std::vector<int64_t> input1{2, 1, 120, 160, 16};
    std::vector<int64_t> output{2, 1, 120, 160, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_0) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{32, 32, 32, 32, 16};
    std::vector<int64_t> input1{32, 32, 16, 16, 16};
    std::vector<int64_t> output{32, 32, 16, 16, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 32 32 32 0 0 32 32 16 16 16384 4096 512 256 1024 1 32 ");
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_1) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 28, 56, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 2 1 1 0 0 28 56 7 8 25088 896 896 128 2 1 56 ");
}

TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_2) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 28, 56, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_3) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 7, 8, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_4) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 7, 8, 16};
    std::vector<int64_t> input1{2, 1, 1, 1, 16};
    std::vector<int64_t> output{2, 1, 1, 1, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_5) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 64, 64, 16};
    std::vector<int64_t> input1{2, 1, 1, 1, 16};
    std::vector<int64_t> output{2, 1, 1, 1, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_6) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 1, 1, 16};
    std::vector<int64_t> input1{2, 1, 7, 8, 16};
    std::vector<int64_t> output{2, 1, 7, 8, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_7) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 1, 3200, 3000, 16};
    std::vector<int64_t> input1{2, 1, 120, 160, 16};
    std::vector<int64_t> output{2, 1, 120, 160, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_8) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1, \"tensor_c\": 2048}}";

    std::vector<int64_t> input0{2, 2, 96, 128, 16};
    std::vector<int64_t> input1{2, 2, 48, 64, 16};
    std::vector<int64_t> output{2, 2, 48, 64, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 32 0 0 3 3 96 128 48 64 196608 49152 2048 1024 4 1 128 ");
}
TEST_F(ResizeBilinearV2GradTiling, resize_bilinear_v2_grad_tiling_9) {
    std::string op_name = "ResizeBilinearV2Grad";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"l1_support\": 1}}";

    std::vector<int64_t> input0{2, 1, 3200, 3000, 16};
    std::vector<int64_t> input1{2, 1, 120, 160, 16};
    std::vector<int64_t> output{2, 1, 120, 160, 16};

    auto opParas = op::ResizeBilinearV2Grad("ResizeBilinearV2Grad");
    TENSOR_INPUT_WITH_SHAPE(opParas, grads, input0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_INPUT_WITH_SHAPE(opParas, original_image, input1, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
