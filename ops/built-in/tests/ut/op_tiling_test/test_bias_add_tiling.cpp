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

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"

using namespace std;

class BiasAddTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BiasAddTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BiasAddTiling TearDown" << std::endl;
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
using namespace ut_util;
/*
.INPUT(x, TensorType::NumberType())
    .INPUT(bias, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
*/

TEST_F(BiasAddTiling, BiasAdd_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [1, 1, 4]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 4 4 ");
}

TEST_F(BiasAddTiling, BiasAdd_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1999, 1999, 4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [1, 1, -1]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3996001 4 12 10407 ");
}

TEST_F(BiasAddTiling, BiasAdd_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1999, 1999, 4},
      {4},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2], "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
    RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3996001 4 12 10407 ");
}


TEST_F(BiasAddTiling, BiasAdd_tiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 1, 1, 16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1, 1, 1, 1, 16},
      {1},
  };
  
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NCDHW, ori_shapes[0], ge::FORMAT_NCDHW, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NCDHW, ori_shapes[0], ge::FORMAT_NCDHW, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NCDHW, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320],"121": [32, 2, 30704, 15344],"210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152],"000": [32, 2, 30704, 15344],"999": [32, 2, 30704, 15344], "200": [32, 2, 30704, 15344]}, "_outs_uint1":false, "_elewise_vars": {"220000000": [10000, 20000, 30000],"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2], "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 1 16 ");
}


TEST_F(BiasAddTiling, BiasAdd_tiling5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 1, 16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1, 1, 1, 16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NCDHW, ori_shapes[0], ge::FORMAT_NCDHW, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NCDHW, ori_shapes[0], ge::FORMAT_NCDHW, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NCDHW, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {16},
      {16},
  };

  vector<vector<int64_t>> ori_shapes = {
      {16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, ori_shapes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, ori_shapes[0], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_ND, ori_shapes[0], ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_ND, ori_shapes[0], ge::FORMAT_ND, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDC1HWC0, ori_shapes[0], ge::FORMAT_NDHWC, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NDC1HWC0, ori_shapes[0], ge::FORMAT_NDHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_ND, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling9) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 1, 1, 1, 16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1, 1, 1, 1, 16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDC1HWC0, ori_shapes[0], ge::FORMAT_NCDHW, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NDC1HWC0, ori_shapes[0], ge::FORMAT_NCDHW, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320],"121": [32, 2, 30704, 15344],"210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152],"000": [32, 2, 30704, 15344],"999": [32, 2, 30704, 15344], "200": [32, 2, 30704, 15344]}, "_outs_uint1":false, "_elewise_vars": {"220000000": [10000, 20000, 30000],"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2], "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 16 ");
}

TEST_F(BiasAddTiling, BiasAdd_tiling10) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NDHWC, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NDHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling11) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NC1HWC0, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NC1HWC0, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling12) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling13) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NCHW, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NC1HWC0, ori_shapes[0], ge::FORMAT_NCHW, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling14) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,1,16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDHWC, ori_shapes[0], ge::FORMAT_NCHW, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NDHWC, ori_shapes[0], ge::FORMAT_NCHW, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDHWC, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling15) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1,1,1,1,16},
      {1},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1,1,1,1,16},
      {1},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDHWC, ori_shapes[0], ge::FORMAT_NCHW, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NDHWC, ori_shapes[0], ge::FORMAT_NCHW, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDHWC, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "210": [32, 2, 43680, 21840]}, "_outs_uint1":false, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2]})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(BiasAddTiling, BiasAdd_tiling16) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("BiasAdd");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::BiasAdd("BiasAdd");

  vector<vector<int64_t>> input_shapes = {
      {1, 1, 1, 1, 16},
      {16},
  };

  vector<vector<int64_t>> ori_shapes = {
      {1, 1, 1, 1, 16},
      {16},
  };

  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, x, input_shapes[0], dtypes[0], ge::FORMAT_NDHWC, ori_shapes[0], ge::FORMAT_NHWC, {});
  TENSOR_INPUT_WITH_ORI_SHAPE(opParas, bias, input_shapes[1], dtypes[1], ge::FORMAT_NDHWC, ori_shapes[0], ge::FORMAT_NHWC, {});

  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, input_shapes[0], ge::DT_FLOAT16, ge::FORMAT_NDHWC, {});
  std::string compileInfo =
      R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320],"121": [32, 2, 30704, 15344],"210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152],"000": [32, 2, 30704, 15344],"999": [32, 2, 30704, 15344], "200": [32, 2, 30704, 15344]}, "_outs_uint1":false, "_elewise_vars": {"220000000": [10000, 20000, 30000],"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "221000000": [10000, 10100], "221000001": [10000, 10100, 20000, 30000], "221000002": [10000, 10100, 20000, 30001], "221000004": [10000, 10100, 20001, 30001]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "221000000": ["_dim_0_0", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_1_0", "_block_factor_1", "_ub_factor_1"]}, "boardcast_bias_shape": [-2], "is_unknown_rank": true})";

  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 16 16 ");
}

