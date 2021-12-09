#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "nn_detect_ops.h"
#include "array_ops.h"

using namespace std;

class RoiAlignGradTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RoiAlignGradTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RoiAlignGradTiling TearDown" << std::endl;
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
.INPUT(ydiff, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(rois_n, TensorType({DT_INT32}))
    .OUTPUT(xdiff, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(xdiff_shape, ListInt)
    .REQUIRED_ATTR(pooled_width, Int)
    .REQUIRED_ATTR(pooled_height, Int)
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(sample_num, Int, 2)
*/

TEST_F(RoiAlignGradTiling, roi_align_grad_tiling_0) {
  std::string op_name = "ROIAlignGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ROIAlignGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32}}";

  std::vector<int64_t> inputA{1024, 16, 14, 14, 16};
  std::vector<int64_t> inputB{1024, 5};
  std::vector<int64_t> inputC{1024};
  std::vector<int64_t> output{2, 16, 24, 40, 16};

  auto opParas = op::ROIAlignGrad("ROIAlignGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, ydiff, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois_n, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, xdiff, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 32 1024 5 16 24 40 ");
}

TEST_F(RoiAlignGradTiling, roi_align_grad_tiling_1) {
  std::string op_name = "ROIAlignGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ROIAlignGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32}}";

  std::vector<int64_t> inputA{1024, 16, 90, 90, 16};
  std::vector<int64_t> inputB{1024, 5};
  std::vector<int64_t> inputC{1024};
  std::vector<int64_t> output{2, 16, 24, 90, 16};

  auto opParas = op::ROIAlignGrad("ROIAlignGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, ydiff, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois_n, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, xdiff, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 1024 5 16 24 90 ");
}

TEST_F(RoiAlignGradTiling, roi_align_grad_tiling_2) {
  std::string op_name = "ROIAlignGrad";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ROIAlignGrad");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32}}";

  std::vector<int64_t> inputA{1024, 8, 90, 90, 16};
  std::vector<int64_t> inputB{1024, 5};
  std::vector<int64_t> inputC{1024};
  std::vector<int64_t> output{2, 16, 24, 40, 16};

  auto opParas = op::ROIAlignGrad("ROIAlignGrad");
  TENSOR_INPUT_WITH_SHAPE(opParas, ydiff, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois, inputB, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, rois_n, inputC, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, xdiff, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 32 1024 5 8 24 40 ");
}
