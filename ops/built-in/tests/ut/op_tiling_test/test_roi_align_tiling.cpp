#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_detect_ops.h"
#include "array_ops.h"

using namespace std;

class RoiAlignTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RoiAlignTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RoiAlignTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;
/*
.INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(rois_n, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(pooled_width, Int)
    .ATTR(sample_num, Int, 2)
    .ATTR(roi_end_mode, Int, 1)
*/

TEST_F(RoiAlignTiling, roi_align_tiling_0) {
  std::string op_name = "ROIAlign";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ROIAlign");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32}}";

  std::vector<int64_t> inputA{2, 16, 24, 40, 16};
  std::vector<int64_t> inputB{1024, 5};
  std::vector<int64_t> inputC{1024};
  std::vector<int64_t> output{1024, 16, 14, 14, 16};

  auto opParas = op::ROIAlign("ROIAlign");
  TENSOR_INPUT_WITH_SHAPE(opParas, features, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois_n, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 1024 5 16 24 40 ");
}
