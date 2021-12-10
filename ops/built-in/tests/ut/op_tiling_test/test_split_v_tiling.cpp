#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "test_common.h"
using namespace ge;
using namespace ut_util;
using namespace std;

class SplitVTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitVTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitVTiling TearDown" << std::endl;
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

TEST_F(SplitVTiling, SplitV_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {1},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 232},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT32, ge::DT_INT32};
  vector<int32_t> SizSplits{1820};
  vector<int32_t> SplitDim{0};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, SizSplits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, SplitDim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":253952, \"num_split\":1}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 32 422240 1820 13195 13195 0 13195 13195 0 13195 13195 232 1 422240 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {232, 1820},
      {3},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {100, 1820},
      {96, 1820},
      {36, 18720},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{100, 96, 36};
  std::vector<int32_t> split_dim{0};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":63488, \"num_split\":3}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 32 422240 232 0 0 0 0 0 0 0 0 1820 1 422240 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {5},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 80}, {1820, 50}, {1820, 1}, {1820, 46}, {1820, 55},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{80, 50, 1, 46, 55};
  std::vector<int32_t> split_dim{-1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":5}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 32 422240 232 0 0 0 0 0 0 0 0 1 1820 232 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1, 48, 512},
      {48},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1, 1, 512},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> split_dim{1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":48}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 32 24576 48 0 0 0 0 0 0 0 0 512 1 24576 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {18720, 3},
      {3},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {18720, 1},
      {18720, 1},
      {18720, 1},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{1, 1, 1};
  std::vector<int32_t> split_dim{-1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":3}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 25 56160 3 0 0 0 0 0 0 0 0 1 18720 3 0 224 3 0 3 1 0 1 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {18725, 6},
      {3},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {18725, 1},
      {18725, 2},
      {18725, 3},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{1, 2, 3};
  std::vector<int32_t> split_dim{-1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":3}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "5 32 112350 6 592 373 0 592 592 0 373 373 1 18725 6 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {48000, 256},
      {7},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {48000, 80}, {48000, 80}, {48000, 80}, {48000, 1}, {48000, 1}, {48000, 1}, {48000, 13},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{80, 80, 80, 1, 1, 1, 13};
  std::vector<int32_t> split_dim{-1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":7}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "6 32 12288000 256 1504 1376 11 96 128 10 96 128 1 48000 256 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {40000, 85},
      {4},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {40000, 2},
      {40000, 2},
      {40000, 1},
      {40000, 80},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{2, 2, 1, 80};
  std::vector<int32_t> split_dim{-1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "7 32 3400000 85 1280 320 5 0 256 1 64 256 1 40000 85 0 0 0 0 0 0 0 0 0 ");
}

TEST_F(SplitVTiling, SplitV_tiling9) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {40000, 85},
      {4},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {40000, 2},
      {40000, 2},
      {40000, 1},
      {40000, 80},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32};
  std::vector<int32_t> size_splits{32, 32, 16, 1280};
  std::vector<int32_t> split_dim{-1};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_FRACTAL_NZ, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_FRACTAL_NZ,
                                          size_splits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_FRACTAL_NZ, split_dim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_FRACTAL_NZ, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), to_string(runInfo.GetAllTilingData()));
}
TEST_F(SplitVTiling, SplitV_tiling10) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitV");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SplitV("SplitV");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {1},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 232},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT32, ge::DT_INT32};
  vector<int32_t> SizSplits{1820};
  vector<int32_t> SplitDim{0};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input_shapes[0], dtypes[0], FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, size_splits, input_shapes[1], dtypes[1], FORMAT_ND, SizSplits);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, split_dim, input_shapes[2], dtypes[2], FORMAT_ND, SplitDim);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output_shapes[i], dtypes[0], FORMAT_ND, {});
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":13185, \"num_split\":1}}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 32 422240 1820 13195 13195 1 42 13153 1 42 13153 232 1 422240 0 0 0 0 0 0 0 0 0 ");
}
