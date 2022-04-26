/*
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <iostream>
#include <vector>
#include <map>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class BiasAddGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BiasAddGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BiasAddGradTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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
#include "test_common.h"
using namespace ut_util;
/*
    .INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
*/

TEST_F(BiasAddGradTiling, BiasAdd_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 8 1 ");
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {1999, 1999, 4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1999 1999 4 63 1 ");
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {1999, 1999, 4},
  };
  vector<int64_t> origin_shape = {4,4,4,4,4};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NDHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1999 1999 4 63 1 ");
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {65536, 2, 16, 16},//{c1*h*w, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {32,128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 128 65536 8192 1 ");
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {65536*2, 2, 16, 16},//{DC1HW, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {32,2,128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NDHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32"})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 128 65536 8192 1 ");
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {2, 4, 128, 128, 2, 16, 16},//{DC1HW, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {32,2,128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z_3D, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NDHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9, 41], "_ub_info": [16256, 16000, 16256, 16256], "_ub_info_rf": [16256, 16000, 16256, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 2 4 524288 16 32768 1016 ");
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {4, 128, 128, 2, 16, 16},//{c1*h*w, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {32,128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 524288 16 16384 1016 ");
}

vector<int64_t> GenTestShapeTrue(const vector<int64_t> origin_shape, const ge::Format origin_format,
                                 const ge::Format format, int num) {
  int64_t n = origin_shape[0], d = 13;
  int64_t c, h, w;
  if (origin_format == ge::FORMAT_NCHW) {
    c = origin_shape[1];
    h = origin_shape[2];
    w = origin_shape[3];
  } else {
    c = origin_shape[3];
    h = origin_shape[1];
    w = origin_shape[2];
  }
  std:map<ge::Format, vector<vector<int64_t>>> test_format_shapes = {
    {ge::FORMAT_FRACTAL_NZ,{{n, c, h, w},{n, (c+15)/16, h, w, 16},{(c+15)/16, h, w, (n+15)/16, 16, 16},{n, c, h, w}}},
    {ge::FORMAT_FRACTAL_Z,{{(c+15)/16, h, w, (n+15)/16, 16, 16}}},
    {ge::FORMAT_FRACTAL_Z_3D,{{d, (c+15)/16, h, w, (n+15)/16, 16, 16}}},
    {ge::FORMAT_NC1HWC0,{{n, (c+15)/16, h, w, 16}}},
    {ge::FORMAT_NDC1HWC0,{{n, d, (c+15)/16, h, w, 16}}},
    {ge::FORMAT_ND,{{n, c, h, w},{n, h, w, c},{n, c}}}
  };
  std::map<ge::Format, vector<vector<int64_t>>>::iterator iter;
  iter = test_format_shapes.find(format);
  if(iter == test_format_shapes.end()) {
    return {0};
  }
  if(num >= iter->second.size()) {
    return {0};
  }
  return iter->second[num];
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<int64_t> origin_shape = {60,80,40,100};
  vector<ge::Format> origin_formats = {ge::FORMAT_NHWC, ge::FORMAT_NCHW};
  vector<ge::Format> input_formats = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_3D,
                                      ge::FORMAT_NC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_ND};
  vector<int> input_nums = {4, 1, 1, 1, 1, 3};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  for (size_t i = 0; i < input_formats.size(); i++) {
    for (size_t j = 0; j < origin_formats.size(); j++) {
      for (int k = 0; k < input_nums[i]; k++) {
        vector<int64_t> shape = GenTestShapeTrue(origin_shape, origin_formats[j], input_formats[i], k);
        TensorDesc tensorInput(ge::Shape(shape),
                               input_formats[i], dtypes[0]);
        tensorInput.SetOriginFormat(origin_formats[j]);
        tensorInput.SetOriginShape(ge::Shape(origin_shape));
        TENSOR_INPUT(opParas, tensorInput, x);
        TENSOR_OUTPUT_WITH_SHAPE(opParas, y, shape, ge::DT_FLOAT16, origin_formats[j], {});
        std::string compileInfo =
            R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9, 41, 20], "_ub_info": [16256, 16000, 16256, 16256, 16256], "_ub_info_rf": [16256, 16000, 16256, 16256, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

        // do tilling, get runInfo
        optiling::utils::OpRunInfo runInfo;
        RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
        std::cout<<to_string(runInfo.GetAllTilingData())<<std::endl;
        //EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 524288 16 16384 1016 ");
      }
    }
  }
}

vector<int64_t> GenTestShapeFalse(const vector<int64_t> origin_shape, const ge::Format origin_format,
                                  const ge::Format format, int num) {
  int64_t n = origin_shape[0], d = 13;
  int64_t c, h, w;
  if (origin_format == ge::FORMAT_NCHW) {
    c = origin_shape[1];
    h = origin_shape[2];
    w = origin_shape[3];
  } else {
    c = origin_shape[3];
    h = origin_shape[1];
    w = origin_shape[2];
  }
  std:map<ge::Format, vector<vector<int64_t>>> test_format_shapes = {
    {ge::FORMAT_FRACTAL_NZ,{{n}}},
    {ge::FORMAT_FRACTAL_Z,{{n, c},{n}}},
    {ge::FORMAT_FRACTAL_Z_3D,{{n, (c+15)/16, h, w, 16},{n, h, w, c}}},
    {ge::FORMAT_NC1HWC0,{{n, c, h, w},{n, h, w, c},{n, (c+15)/16, h, w, 16, 16}}},
    {ge::FORMAT_NDC1HWC0,{{n, c, h, w},{n, h, w, c},{n, d, (c+15)/16, h, w, 16, 16},{0}}},
    {ge::FORMAT_ND,{{n},{0}}}
  };
  std::map<ge::Format, vector<vector<int64_t>>>::iterator iter;
  iter = test_format_shapes.find(format);
  if(iter == test_format_shapes.end()) {
    return {0};
  }
  if(num >= iter->second.size()) {
    return {0};
  }
  return iter->second[num];
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling9) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<int64_t> origin_shape = {60,80,40,100};
  vector<ge::Format> origin_formats = {ge::FORMAT_NHWC, ge::FORMAT_NCHW};
  vector<ge::Format> input_formats = {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_3D,
                                      ge::FORMAT_NC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_ND};
  vector<int> input_nums = {1, 2, 2, 3, 4, 4};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  for (size_t i = 0; i < input_formats.size(); i++) {
    for (size_t j = 0; j < origin_formats.size(); j++) {
      for (int k = 0; k < input_nums[i]; k++) {
        vector<int64_t> shape = GenTestShapeFalse(origin_shape, origin_formats[j], input_formats[i], k);
        TensorDesc tensorInput(ge::Shape(shape),
                               input_formats[i], dtypes[0]);
        tensorInput.SetOriginFormat(origin_formats[j]);
        tensorInput.SetOriginShape(ge::Shape(origin_shape));
        TENSOR_INPUT(opParas, tensorInput, x);
        TENSOR_OUTPUT_WITH_SHAPE(opParas, y, shape, ge::DT_FLOAT16, origin_formats[j], {});
        std::string compileInfo =
            R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9, 41, 20], "_ub_info": [16256, 16000, 16256, 16256, 16256], "_ub_info_rf": [16256, 16000, 16256, 16256, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

        // do tilling, get runInfo
        optiling::utils::OpRunInfo runInfo;
        std::cout<<"RUN_TILING_V3_FALSE:ijk="<<i<<j<<k<<std::endl;
        RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
      }
    }
  }
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling10) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {2, 16, 16},//{c1*h*w, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_NZ, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling11) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {4, 128, 128, 2, 16, 16},//{c1*h*w, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {4, 128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z_3D, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NDHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z_3D, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9], "_ub_info": [16256, 16000, 16256], "_ub_info_rf": [16256, 16000, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling12) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {2, 4, 128, 128, 2, 16, 16},//{c1*h*w, n1, n0, c0}
  };
  vector<int64_t> origin_shape = {4, 2, 128,128,64};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_Z_3D, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NDHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_FRACTAL_Z_3D, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9, 41, 20], "_ub_info": [16256, 16000, 16256, 16256, 16256], "_ub_info_rf": [16256, 16000, 16256, 16256, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling13) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {3, 17, 1, 101, 61},//{NCDHW}
  };
  vector<int64_t> origin_shape = {3, 17, 1, 101, 61};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_NZ, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NCDHW);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NCDHW, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9, 41, 20], "_ub_info": [16256, 16000, 16256, 16256, 16256], "_ub_info_rf": [16256, 16000, 16256, 16256, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddGradTiling, BiasAdd_tiling14) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAddGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAddGrad("BiasAddGrad");

  vector<vector<int64_t>> input_shapes = {
      {3, 1, 17, 101, 61},//{FORMAT_NDHWC}
  };
  vector<int64_t> origin_shape = {3, 1, 17, 101, 61};

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  TensorDesc tensorInput(ge::Shape(input_shapes[0]), ge::FORMAT_FRACTAL_NZ, dtypes[0]);
  tensorInput.SetOriginFormat(ge::FORMAT_NDHWC);
  tensorInput.SetOriginShape(ge::Shape(origin_shape));
  TENSOR_INPUT(opParas, tensorInput, x);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDHWC, {});
  std::string compileInfo =
      R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5, 4, 9, 41, 20], "_ub_info": [16256, 16000, 16256, 16256, 16256], "_ub_info_rf": [16256, 16000, 16256, 16256, 16256], "reduce_mean_cof_dtype": "float32", "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
}
