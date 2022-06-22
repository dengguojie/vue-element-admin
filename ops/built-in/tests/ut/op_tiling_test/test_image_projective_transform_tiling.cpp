/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * image_projective_transform_v2 tiling ut case
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <graph/utils/type_utils.h>
#define private public
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_util.h"
#include "array_ops.h"
#include "image_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;

class ImageProjectiveTransformTiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "ImageProjectiveTransformTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ImageProjectiveTransformTiling TearDown" << std::endl;
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

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_1) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{1, 5, 3, 1};
    std::vector<int64_t> input1{1, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{1, 5, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{5, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 1 5 3 1 15 5 3 21162 0 5 0 1 1 ");
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_2) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{2, 5, 3, 1};
    std::vector<int64_t> input1{2, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{2, 5, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{5, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 2 5 3 1 15 5 3 21162 0 5 0 1 2 ");
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_3) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{33, 5, 3, 1};
    std::vector<int64_t> input1{1, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{33, 5, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{5, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 32 33 5 3 1 15 5 3 21162 0 5 1 2 1 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_4) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{33, 5, 3, 1};
    std::vector<int64_t> input1{33, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{33, 5, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{5, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 32 33 5 3 1 15 5 3 21162 0 5 1 2 1 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_5) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{1, 500, 300, 1};
    std::vector<int64_t> input1{1, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{1, 500, 300, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{500, 300};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 1 1 500 300 1 150000 500 300 211 2 78 0 1 1 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_6) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{2, 500, 300, 1};
    std::vector<int64_t> input1{2, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{2, 500, 300, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{500, 300};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 2 2 500 300 1 150000 500 300 211 2 78 0 1 2 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_7) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{33, 500, 300, 1};
    std::vector<int64_t> input1{1, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{33, 500, 300, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{500, 300};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 32 33 500 300 1 150000 500 300 211 2 78 1 2 1 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_8) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{33, 500, 300, 1};
    std::vector<int64_t> input1{33, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{33, 500, 300, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{500, 300};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "7 32 33 500 300 1 150000 500 300 211 2 78 1 2 1 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_9) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{1, 2, 3, 1};
    std::vector<int64_t> input1{1, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{1, 2, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{2, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 1 1 2 3 1 6 2 3 21162 0 2 0 1 1 ");
}


TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_10) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{3, 2, 3, 1};
    std::vector<int64_t> input1{3, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{3, 2, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int32_t> output_data{2, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "9 3 3 2 3 1 6 2 3 21162 0 2 0 1 3 ");
}

TEST_F(ImageProjectiveTransformTiling, image_projective_transform_tiling_11) {
    using namespace ut_util;
    std::string op_name = "ImageProjectiveTransform";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"ub_ele\": 63488, \"core_num\": 32, \"trans_dtype_size\": 4, \"block_byte_size\": 32}}";

    std::vector<int64_t> input0{1, 5, 3, 1};
    std::vector<int64_t> input1{1, 8};
    std::vector<int64_t> input2{2};
    std::vector<int64_t> output{1, 5, 3, 1};

    std::string format1 = "NHWC";
    std::string format2 = "ND";

    vector<ge::DataType> dtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16};
    std::vector<int32_t> output_data{5, 3};

    auto opParas = op::ImageProjectiveTransform("ImageProjectiveTransform");
    TENSOR_INPUT_WITH_SHAPE(opParas, images, input0, dtype[0], TypeUtils::SerialStringToFormat(format1), {});
    TENSOR_INPUT_WITH_SHAPE(opParas, transforms, input1, dtype[1], TypeUtils::SerialStringToFormat(format2), {});
    TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, output_shape, input2, dtype[2], TypeUtils::SerialStringToFormat(format2), output_data);
    TENSOR_OUTPUT_WITH_SHAPE(opParas, var, output, dtype[3], TypeUtils::SerialStringToFormat(format1), {});

    optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
    EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 1 1 5 3 1 15 5 3 21162 0 5 0 1 1 ");
}
