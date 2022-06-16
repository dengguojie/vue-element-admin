//
// Created by wangqi on 2022/05/14
//

#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/elewise_v3.h"
#include "op_tiling/tiling_handler.h"

#include "common_autotiling_util.h"
#include "graph/utils/attr_utils.h"

#define private public
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

using namespace optiling;
class ElewiseTilingRT2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ElewiseTilingRT2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ElewiseTilingRT2 TearDown" << std::endl;
  }
};

TEST_F(ElewiseTilingRT2, empty_mode_none_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{2, 0, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{2, 0, 2}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(ElewiseTilingRT2, empty_mode_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{2, 0, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{2, 0, 2}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(ElewiseTilingRT2, common_none_custom_tiling_multicore) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 16384, 8192}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "268435456, 8388608, 8192";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(ElewiseTilingRT2, common_custom_tiling_multicore) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 16384, 8192}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "268435456, 8388608, 8192";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(ElewiseTilingRT2, common_none_custom_tiling_single_core) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{256},{256}};
  std::vector<std::vector<int64_t>> out_shapes = {{256}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = true;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}},
                                          {"230", {32, 4, 21840, 10920}},
                                          {"320", {32, 4, 21840, 10920}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }},
                                             {"223000000", {10000, 20000, 30000 }},
                                             {"223010000", {10000, 20000, 30000 }},
                                             {"232000000", {10000, 20000, 30000 }},
                                             {"232010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "256, 256, 256";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(ElewiseTilingRT2, common_custom_tiling_single_core) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{256},{256}};
  std::vector<std::vector<int64_t>> out_shapes = {{256}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = true;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}},
                                          {"230", {32, 4, 21840, 10920}},
                                          {"320", {32, 4, 21840, 10920}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }},
                                             {"223000000", {10000, 20000, 30000 }},
                                             {"223010000", {10000, 20000, 30000 }},
                                             {"232000000", {10000, 20000, 30000 }},
                                             {"232010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "256, 256, 256";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(ElewiseTilingRT2, only_const_tiling_none_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1, 0, 128, 0, 128, 0";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, only_const_tiling_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "1, 0, 128, 0, 128, 0";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, const_pattern_none_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = true;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = true;
  expect_compile_info.const_block_dims.second = {8, 8};
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, const_pattern_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = true;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = true;
  expect_compile_info.const_block_dims.second = {8, 8};
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, common_relu_fp32) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{275614, 11}};
  std::vector<std::vector<int64_t>> out_shapes = {{275614, 11}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 32760, 16376}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "3031754, 94848, 15872";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(ElewiseTilingRT2, broadcast_scalar_custom_tiling_pattern) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{22, 16384, 4}, {1}};
  std::vector<std::vector<int64_t>> out_shapes = {{22, 16384, 4}};
  ge::DataType in_dtypes = ge::DT_FLOAT16;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = true;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 2, 43680, 21840}}, {"230", {32, 4, 43680, 21840}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {20000, 30000 }},
                                             {"210010000", {20000, 30000 }},
                                             {"223000000", {10000, 20000, 30000 }},
                                             {"223010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1441792, 45056, 15104";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}


TEST_F(ElewiseTilingRT2, scalar_broadcast_none_custom_tiling_pattern) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1, 1}, {16, 7, 4, 35}};
  std::vector<std::vector<int64_t>> out_shapes = {{16, 7, 4, 35}};
  ge::DataType in_dtypes = ge::DT_INT32;
  ge::DataType out_dtypes = ge::DT_INT32;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = true;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}}, {"320", {32, 4, 21832, 10912}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {20000, 30000 }},
                                             {"210010000", {20000, 30000 }},
                                             {"232000000", {10001, 20000, 30000 }},
                                             {"232010000", {10001, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "15680, 512, 512";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 31);
}

TEST_F(ElewiseTilingRT2, broadcast_addcmul_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{275614, 11}, {1, 1}, {275614, 11}, {275614, 11}};
  std::vector<std::vector<int64_t>> out_shapes = {{275614, 11}};
  ge::DataType in_dtypes = ge::DT_FLOAT16;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 4;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"200", {32, 2, 26176, 13088}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"220000000", {10000, 10002, 10003, 20000, 30000}},
                                             {"220010000", {10000, 10002, 10003, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "3031754, 3031754, 3031754, 94848, 11904";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(ElewiseTilingRT2, apply_adam_d_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33},
                                                 {1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33},
                                                 {1024, 256, 33}, {1024, 256, 33}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33}};
  ge::DataType in_dtypes = ge::DT_FLOAT16;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 4;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true; 
  expect_compile_info.base_info.second = {{"100", {32, 2, 16320, 8160}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "8650752, 270336, 8064";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(ElewiseTilingRT2, fuse_mul_add_n_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 364}, {1024, 364}, {1}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 364}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 16384, 8192}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "372736, 11648, 11648";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(ElewiseTilingRT2, ub_bound_limit_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{16384, 1115}};
  std::vector<std::vector<int64_t>> out_shapes = {{16384, 1115}};
  ge::DataType in_dtypes = ge::DT_FLOAT16;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 2, 16384, 8192}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210000000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ElewiseTilingRT2, elewise_var_key_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{275614, 11}};
  std::vector<std::vector<int64_t>> out_shapes = {{275614, 11}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 32760, 16376}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000001", {10000, 20000, 30000}},
                                             {"210000002", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ElewiseTilingRT2, low_shape_same_none_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 256}, {1, 123}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 256}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 32760, 16376}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ElewiseTilingRT2, low_shape_same_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 256}, {1, 123}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 256}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 32760, 16376}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ElewiseTilingRT2, diff_lens_higher_shape_all_one_none_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 256}, {256}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 256}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = true;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}},
                                          {"230", {32, 4, 21840, 10920}},
                                          {"320", {32, 4, 21840, 10920}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}},
                                             {"223000000", {10000, 20000, 30000}},
                                             {"223010000", {10000, 20000, 30000}},
                                             {"232000000", {10000, 20000, 30000}},
                                             {"232010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ElewiseTilingRT2, diff_lens_higher_shape_all_one_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 256}, {256}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 256}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = true;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}},
                                          {"230", {32, 4, 21840, 10920}},
                                          {"320", {32, 4, 21840, 10920}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}},
                                             {"223000000", {10000, 20000, 30000}},
                                             {"223010000", {10000, 20000, 30000}},
                                             {"232000000", {10000, 20000, 30000}},
                                             {"232010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ElewiseTilingRT2, apply_rms_prop_d_st_fail_case) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 16}, {1, 16}, {1, 16}, {1}, {1, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 16}, {1, 16}, {1, 16}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;

  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 4;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 10904, 5448}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210000000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "16, 16, 16";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(ElewiseTilingRT2, dynamic_add_const_elewise_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = true;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = true;
  expect_compile_info.const_block_dims.second = {8, 8};
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, dynamic_add_const_elewise_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  ge::DataType in_dtypes = {ge::DT_FLOAT};
  ge::DataType out_dtypes = {ge::DT_FLOAT};
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = true;
  expect_compile_info.support_broadcast = true;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = true;
  expect_compile_info.const_block_dims.second = {8, 8};
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;
  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, dynamic_cast_s32_to_s64_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  ge::DataType in_dtypes = {ge::DT_INT32};
  ge::DataType out_dtypes = {ge::DT_INT64};
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 256;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 8, 8188, 4092}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}}};
  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1024, 256, 256";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 4);
}

TEST_F(ElewiseTilingRT2, dynamic_cast_s64_to_s32_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  ge::DataType in_dtypes = {ge::DT_INT64};
  ge::DataType out_dtypes = {ge::DT_INT32};
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 8, 8188, 4092}}}; 
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000}},
                                             {"210010000", {10000, 20000, 30000}}};

  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1024, 128, 128";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 8);
}

TEST_F(ElewiseTilingRT2, const_vcmp_support_b64_case) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{11323, 128}, {11323, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{11323, 128}};
  ge::DataType in_dtypes = ge::DT_INT64;
  ge::DataType out_dtypes = ge::DT_INT8;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = true;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = true;
  expect_compile_info.const_block_dims.second = {32, 32};
  expect_compile_info.base_info.first = false;
  expect_compile_info.elewise_vars.first = false;

  test.SetCompileInfo(&expect_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

// test var attr case
TEST_F(ElewiseTilingRT2, elewise_set_attr_case1) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  ge::DataType in_dtypes = ge::DT_FLOAT;
  ge::DataType out_dtypes = ge::DT_FLOAT16;
  AutoTilingTest test(in_shapes, out_shapes, in_dtypes, out_dtypes);
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"100", {32, 4, 16384, 8192}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }}};
  test.SetCompileInfo(&expect_compile_info);
  std::vector<std::pair<std::string, int64_t>> common_attr = {{"alpha", 123}};
  test.SetAttrs<int64_t>(common_attr);

  EXPECT_EQ(test.Test(), true);
}

// Test elewise not all fuse
TEST_F(ElewiseTilingRT2, dynamic_not_all_fuse) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{128, 16, 16, 1}, {128, 16, 16, 1}};
  std::vector<std::vector<int64_t>> in_shapes = {{128, 1, 16, 16, 16}, {128, 1, 16, 16, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{128, 16, 16, 1}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 1, 16, 16, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT8};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NHWC};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NHWC};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 3;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}},
                                          {"111", {32, 4, 21840, 10920}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }},
                                             {"211100000", {40300, 40301, 10000, 10100, 10200, 20000, 30000 }},
                                             {"211100001", {40300, 40301, 10000, 10100, 10200, 20000, 30001 }},
                                             {"211100002", {40300, 40301, 10000, 10100, 10200, 20000, 30002 }},
                                             {"211100003", {40300, 40301, 10000, 10100, 10200, 20000, 30003 }},
                                             {"211110000", {40300, 40301, 10000, 10100, 10200, 20000, 30000 }},
                                             {"211110001", {40300, 40301, 10000, 10100, 10200, 20000, 30001 }},
                                             {"211110002", {40300, 40301, 10000, 10100, 10200, 20000, 30002 }},
                                             {"211110003", {40300, 40301, 10000, 10100, 10200, 20000, 30003 }},};

  test.SetCompileInfo(&expect_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1, 1, 128, 1, 256, 2, 2";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

// Test elewise not all fuse custom
TEST_F(ElewiseTilingRT2, dynamic_not_all_fuse_custom) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{128, 1, 16, 16}, {128, 1, 16, 16}};
  std::vector<std::vector<int64_t>> in_shapes = {{128, 1, 16, 16, 16}, {128, 1, 16, 16, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{128, 1, 16, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 1, 16, 16, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT8};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 1;
  expect_compile_info.base_info.second = {{"100", {32, 4, 21840, 10920}},
                                          {"111", {32, 4, 21840, 10920}}};
  expect_compile_info.elewise_vars.first = true;
  expect_compile_info.elewise_vars.second = {{"210000000", {10000, 20000, 30000 }},
                                             {"210010000", {10000, 20000, 30000 }},
                                             {"211100000", {40100, 40101, 10000, 10100, 10200, 20000, 30000 }},
                                             {"211100001", {40100, 40101, 10000, 10100, 10200, 20000, 30001 }},
                                             {"211100002", {40100, 40101, 10000, 10100, 10200, 20000, 30002 }},
                                             {"211100003", {40100, 40101, 10000, 10100, 10200, 20000, 30003 }},
                                             {"211110000", {40100, 40101, 10000, 10100, 10200, 20000, 30000 }},
                                             {"211110001", {40100, 40101, 10000, 10100, 10200, 20000, 30001 }},
                                             {"211110002", {40100, 40101, 10000, 10100, 10200, 20000, 30002 }},
                                             {"211110003", {40100, 40101, 10000, 10100, 10200, 20000, 30003 }},};

  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes[0]);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "1, 1, 128, 1, 256, 2, 2";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

// Test elewise not all fuse const non custom
TEST_F(ElewiseTilingRT2, only_const_tiling_not_all_fuse) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{128, 1, 16, 16}, {128, 1, 16, 16}};
  std::vector<std::vector<int64_t>> in_shapes = {{128, 1, 16, 16, 16}, {128, 1, 16, 16, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{128, 1, 16, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 1, 16, 16, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT8};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 1;
  expect_compile_info.elewise_vars.first = false;

  test.SetCompileInfo(&expect_compile_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1, 0, 2, 0, 2, 0";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

// Test elewise not all fuse const custom
TEST_F(ElewiseTilingRT2, only_const_tiling_not_all_fuse_custom) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{128, 1, 16, 16}, {128, 1, 16, 16}};
  std::vector<std::vector<int64_t>> in_shapes = {{128, 1, 16, 16, 16}, {128, 1, 16, 16, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{128, 1, 16, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 1, 16, 16, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT8};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 1;
  expect_compile_info.elewise_vars.first = false;

  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes[0]);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "1, 0, 2, 0, 2, 0";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

// eleiwse 5hd not full core
TEST_F(ElewiseTilingRT2, elewise_5hd_not_all_core) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{1, 5, 10, 16}, {1, 5, 10, 16}};
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1, 10, 16, 16}, {1, 1, 10, 16, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{1, 5, 10, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1, 10, 16, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 1;
  expect_compile_info.elewise_vars.first = false;

  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes[0]);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "1, 0, 1, 3, 8, 0";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 20);
}


// eleiwse 5hd single core
TEST_F(ElewiseTilingRT2, elewise_5hd_single_core) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{1, 1, 2, 2}, {1, 1, 2, 2}};
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1, 2, 2, 16}, {1, 1, 2, 2, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{1, 1, 2, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1, 2, 2, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 1;
  expect_compile_info.elewise_vars.first = false;

  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes[0]);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "0, 0, 1, 0, 1, 0";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

// eleiwse 5hd double_buffer core
TEST_F(ElewiseTilingRT2, elewise_5hd_double_buffer) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> ori_in_shapes = {{12345678, 1, 1, 1}, {12345678, 1, 1, 1}};
  std::vector<std::vector<int64_t>> in_shapes = {{12345678, 1, 1, 1, 16}, {12345678, 1, 1, 1, 16}};
  std::vector<std::vector<int64_t>> ori_out_shapes = {{12345678, 1, 1, 1}};
  std::vector<std::vector<int64_t>> out_shapes = {{12345678, 1, 1, 1, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> input_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> input_format = {ge::Format::FORMAT_NC1HWC0};
  std::vector<ge::Format> output_ori_format = {ge::Format::FORMAT_NCHW};
  std::vector<ge::Format> output_format = {ge::Format::FORMAT_NC1HWC0};

  AutoTilingTest test(ori_in_shapes, in_shapes, ori_out_shapes, out_shapes, in_dtypes, out_dtypes, input_ori_format, input_format, output_ori_format, output_format);
  // Construct compile_info
  optiling::v3::ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.pattern = SchPattern::ELETWISE;
  expect_compile_info.classify_inputs_num = 2;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 4, 21840, 10920}}};
  expect_compile_info.contains_need_pad_compute = true;
  expect_compile_info.elewise_fused_index.first = true;
  expect_compile_info.elewise_fused_index.second = {{0}, {1}, {2, 3}, {4}};
  expect_compile_info.elewise_pad_axis.first = true;
  expect_compile_info.elewise_pad_axis.second = 1;
  expect_compile_info.elewise_vars.first = false;

  test.SetCompileInfo(&expect_compile_info);
  OpInfo op_info(&expect_compile_info);
  op_info.SetInputShape(&in_shapes);
  op_info.SetInputType(&in_dtypes[0]);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "1, 0, 566, 0, 682, 1";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}