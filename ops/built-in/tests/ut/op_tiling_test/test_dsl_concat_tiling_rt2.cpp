#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/concat_dsl.h"
#include "op_tiling/tiling_handler.h"

#include "common_autotiling_util.h"

using namespace optiling;
class ConcatDslTilingRt2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatDslTilingRt2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatDslTilingRt2 TearDown" << std::endl;
  }
};

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case1) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{100, 200, 30}, {15, 200, 30}};
  std::vector<std::vector<int64_t>> outputs{{116, 200, 30}};
  std::string compile_info =
      R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "600000, 90000, 1, 1, 21568, 17664";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 4000001);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case2) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{2, 81, 49}, {4, 81, 49}};
  std::vector<std::vector<int64_t>> outputs{{6, 81, 49}};
  std::string compile_info =
      R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "7938, 15876, 1, 752, 424";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000001);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case3) {
  // concat 0 axis, original axis is 0
  std::vector<std::vector<int64_t>> inputs{
      {2, 81, 0},
      {4, 0, 49},
  };
  std::vector<std::vector<int64_t>> outputs{{6, 0, 0}};
  std::string compile_info =
      R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetTilingKey(), 2147483647);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case4) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{2, 81, 49}, {4, 81, 49}};
  std::vector<std::vector<int64_t>> outputs{{6, 81, 49}};
  std::string compile_info =
      R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_INT8;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "7938, 15876, 1, 768, 288";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000001);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case5) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{2, 81, 49}, {2, 81, 49}};
  std::vector<std::vector<int64_t>> outputs{{2, 81, 98}};
  std::string compile_info =
      R"({"_ori_axis": 2, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "49, 49, 1, 168";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetTilingKey(), 2100000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case6) {
  // concat 1 axis, original axis is 2
  std::vector<std::vector<int64_t>> inputs{};
  int64_t input_numbers = 48;
  for (int64_t i = 0; i < input_numbers; i++) {
    inputs.push_back({128, 84, 1});
  }
  std::vector<std::vector<int64_t>> outputs{{128, 84, 48}};
  std::string compile_info =
      R"({"_ori_axis": 2, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1, 1, 1, 640";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 17);
  EXPECT_EQ(test.GetTilingKey(), 6000000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case7) {
  // concat 1 axis, original axis is 2
  std::vector<std::vector<int64_t>> inputs{};
  int64_t input_numbers = 24;
  for (int64_t i = 0; i < input_numbers; i++) {
    inputs.push_back({128, 84, 1});
  }
  std::vector<std::vector<int64_t>> outputs{{128, 84, 24}};
  std::string compile_info =
      R"({"_ori_axis": 2, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1, 1, 1, 1280";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 9);
  EXPECT_EQ(test.GetTilingKey(), 6000000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case8) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{4, 28}, {4, 28}};
  std::vector<std::vector<int64_t>> outputs{{4, 56}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "28, 28";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetTilingKey(), 0);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case9) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{40, 281}, {40, 280}};
  std::vector<std::vector<int64_t>> outputs{{40, 561}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[false, true], [false, true]], "_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000001": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "281, 280, 2, 561, 288";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 20);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case10) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{320, 288}, {320, 192}, {320, 64}};
  std::vector<std::vector<int64_t>> outputs{{320, 544}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "320, 288, 192, 1, 60";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 6);
  EXPECT_EQ(test.GetTilingKey(), 4100000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case11) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{4480, 9880}, {4480, 45920}, {4480, 64}};
  std::vector<std::vector<int64_t>> outputs{{4480, 55864}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "4480, 9880, 45920, 9, 16, 2048, 27008, 8064";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 4000000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case12) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{4480, 745}, {4480, 328}};
  std::vector<std::vector<int64_t>> outputs{{4480, 1073}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "4480, 745, 328, 2, 128, 240, 8, 8, 4096";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 18);
  EXPECT_EQ(test.GetTilingKey(), 2000000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case13) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{1, 7, 16, 64}, {1, 7, 16, 433}};
  std::vector<std::vector<int64_t>> outputs{{1, 7, 16, 497}};
  std::string compile_info =
      R"({"_ori_axis": 3, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "112, 64, 433, 1, 112, 224, 1, 16, 7168";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 3);
  EXPECT_EQ(test.GetTilingKey(), 2000001);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case14) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{81, 16, 104}, {81, 16, 792}};
  std::vector<std::vector<int64_t>> outputs{{81, 16, 896}};
  std::string compile_info =
      R"({"_ori_axis": 2, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1296, 104, 792, 1, 64";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 21);
  EXPECT_EQ(test.GetTilingKey(), 5100000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case15) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{81, 16, 104}, {81, 16, 792}};
  std::vector<std::vector<int64_t>> outputs{{81, 16, 896}};
  std::string compile_info =
      R"({"_ori_axis": 2, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  std::vector<int64_t> ori_axis{2};

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);
  OpInfo opInfo(&concat_info);
  opInfo.SetAxes(&ori_axis);
  opInfo.SetInputShape(&inputs);

  EXPECT_EQ(test.Test(&opInfo), true);
  std::string expect_tiling_data = "1296, 104, 792, 1, 64";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 21);
  EXPECT_EQ(test.GetTilingKey(), 5100000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case16) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{32, 1024}, {32, 3}};
  std::vector<std::vector<int64_t>> outputs{{32, 1027}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_INT64;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "32, 1024, 3, 1, 32, 248, 4, 1, 1024";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 5);
  EXPECT_EQ(test.GetTilingKey(), 2000001);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case17) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{3280, 480}, {3280, 246}};
  std::vector<std::vector<int64_t>> outputs{{3280, 726}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, true], [false, true]], "_align_vars": [0, 1], "_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_1", "_offset_1"], "3000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_1", "_offset_1"], "4000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1"], "4100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_1"], "2000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "2100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5000001": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_offset_1"], "5100000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_dim_0_1", "_dim_1_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  ge::DataType dtype = ge::DT_INT32;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "3280, 480, 246, 1, 128, 232, 8, 8, 2048";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 26);
  EXPECT_EQ(test.GetTilingKey(), 2000000);
}

TEST_F(ConcatDslTilingRt2, concat_dsl_tiling_case18) {
  // concat 1 axis, original axis is 1
  std::vector<std::vector<int64_t>> inputs{{3280, 480}, {3280, 778}};
  std::vector<std::vector<int64_t>> outputs{{3280, 1258}};
  std::string compile_info =
      R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": true, "_is_const": false})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::concat::ConcatCompileInfo concat_info;
  test.SetCompileInfo(compile_info, &concat_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1, 0, 0, 8, 0, 1, 128, 240";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 26);
}
