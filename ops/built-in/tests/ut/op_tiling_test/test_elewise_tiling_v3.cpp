//
// Created by wangqi on 2021/10/30.
//

#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"

#include "array_ops.h"
#include "op_tiling/elewise_v3.h"

using namespace std;
using namespace ge;
using namespace optiling;
using namespace v3;

class ElewiseTilingV3 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ElewiseTilingV3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ElewiseTilingV3 TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy_s(&tmp, sizeof(&tmp), data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

static void ConstructData(ge::OpDescPtr& op_desc,
                          const std::vector<int64_t>& shape,
                          const ge::DataType dtype,
                          const ge::Format format,
                          const std::string& params_type) {
  ge:GeTensorDesc tensor;
  tensor.SetShape(ge::GeShape(shape));
  tensor.SetDataType(dtype);
  tensor.SetFormat(format);
  if (params_type == "INPUT") {
    op_desc->AddInputDesc(tensor);
  } else if (params_type == "OUTPUT") {
    op_desc->AddOutputDesc(tensor);
  } else {
    std::cout << "The enter data type is wrong, only support INPUT and OUTPUT" << std::endl;
  }
}

static ge::Operator ConstructOpParas(const std::vector<std::vector<int64_t>>& input_shapes,
                                     const std::vector<std::vector<int64_t>>& output_shapes,
                                     const std::vector<ge::DataType>& input_dtypes,
                                     const std::vector<ge::DataType>& output_dtypes,
                                     const std::vector<ge::Format>& input_formats,
                                     const std::vector<ge::Format>& output_formats) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (uint32_t i = 0; i < input_shapes.size(); i++) {
    ConstructData(op_desc, input_shapes[i], input_dtypes[i], input_formats[i], "INPUT");
  }
  for (uint32_t i = 0; i < output_shapes.size(); i++) {
    ConstructData(op_desc, output_shapes[i], output_dtypes[i], output_formats[i], "OUTPUT");
  }
  return ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
}

template<typename T1, typename T2>
static bool CompareMap(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
    std::cout << "map size wrong!" << std::endl;
    return false;
  }
  for (const auto& it : map1) {
    if (map2.count(it.first) == 0) {
      std::cout << "map key not match" << std::endl;
      return false;
    }
    if (map1.at(it.first) != map2.at(it.first)) {
      std::cout << "map value at key: " << it.first << " is not equal" << std::endl;
      return false;
    }
  }
  return true;
}

static bool CompareCompileInfo1(const ElewiseCompileInfo& expect_compile_info,
                                const ElewiseCompileInfo& real_compile_info) {
  // outs_uint1
  if (expect_compile_info.has_outs_uint1 != real_compile_info.has_outs_uint1) {
    std::cout << "The has_outs_uint1 is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.outs_uint1 != real_compile_info.outs_uint1) {
    std::cout << "The outs_uint1 is wrong" << std::endl;
    return false;
  }
  // flag_info
  if (expect_compile_info.has_flag_info != real_compile_info.has_flag_info) {
    std::cout << "The has_flag_info is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.flag_size != real_compile_info.flag_size) {
    std::cout << "The flag_size is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.only_const_tiling != real_compile_info.only_const_tiling) {
    std::cout << "The only_const_tiling is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.is_const_shapes != real_compile_info.is_const_shapes) {
    std::cout << "The is_const_shapes is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.use_special_pattern != real_compile_info.use_special_pattern) {
    std::cout << "The use_special_pattern is wrong" << std::endl;
    return false;
  }
  return true;
}

static bool CompareCompileInfo2(const ElewiseCompileInfo& expect_compile_info,
                                const ElewiseCompileInfo& real_compile_info) {
  // base_info
  if (expect_compile_info.pattern_key != real_compile_info.pattern_key) {
    std::cout << "The pattern_key is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.core_num != real_compile_info.core_num) {
    std::cout << "The core_num is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.max_dtype != real_compile_info.max_dtype) {
    std::cout << "The max_dtype is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.max_available_ub != real_compile_info.max_available_ub) {
    std::cout << "The max_available_ub is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.max_available_ub_db != real_compile_info.max_available_ub_db) {
    std::cout << "The max_available_ub_db is wrong" << std::endl;
    return false;
  }
  // const_block_dims
  if (expect_compile_info.const_block_dims != real_compile_info.const_block_dims) {
    std::cout << "The const block dims is not match" << std::endl;
    return false;
  }
  // elewise_vars_size
  if (expect_compile_info.elewise_vars_size != real_compile_info.elewise_vars_size) {
    std::cout << "The elewise_vars_size is not match" << std::endl;
    return false;
  }
  // broadcast_pattern
  if (expect_compile_info.broadcast_pattern != real_compile_info.broadcast_pattern) {
    std::cout << "The broadcast_pattern is not match" << std::endl;
    return false;
  }
  return true;
}

static bool CompareElewiseCompieInfo(const ElewiseCompileInfo& expect_compile_info,
                                     const ElewiseCompileInfo& real_compile_info) {
  bool ret = true;
  ret = ret && CompareCompileInfo1(expect_compile_info, real_compile_info);
  ret = ret && CompareCompileInfo2(expect_compile_info, real_compile_info);
  return ret;
}

// Test CreateElewiseTilingHandler
TEST_F(ElewiseTilingV3, elewise_handler1) {
  std::string compile_info_in = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false], "_base_info": {"100": [32, 4, 16384, 8192]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ], "210010000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
  nlohmann::json op_info = nlohmann::json::parse(compile_info_in.c_str());
  auto parsed_ptr =
    std::static_pointer_cast<ElewiseTilingHandler>(CreateElewiseTilingHandler("elewise_handler1", "ElemWise", op_info));
  ASSERT_TRUE(parsed_ptr);
}

TEST_F(ElewiseTilingV3, ConstructTest1) {
  std::string compile_info_in = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "_flag_info": [false, false, false, true, false, false], "_base_info": {"100": [32, 4, 16384, 8192]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ], "210010000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
  nlohmann::json op_info = nlohmann::json::parse(compile_info_in.c_str());
  ElewiseCompileInfo actual_compile_info("ElemWise", op_info);
  ElewiseCompileInfo expect_compile_info;
  expect_compile_info.has_outs_uint1 = true;
  expect_compile_info.outs_uint1 = false;
  expect_compile_info.has_flag_info = true;
  expect_compile_info.flag_size = 6;
  expect_compile_info.only_const_tiling = false;
  expect_compile_info.is_const_shapes = false;
  expect_compile_info.use_special_pattern = true;
  expect_compile_info.pattern_key = 1;
  expect_compile_info.core_num = 32;
  expect_compile_info.max_dtype = 4;
  expect_compile_info.max_available_ub = 16384;
  expect_compile_info.max_available_ub_db = 8192;
  expect_compile_info.const_block_dims = -1;
  expect_compile_info.elewise_vars_size = 3;
  expect_compile_info.broadcast_pattern = false;
  ASSERT_TRUE(CompareElewiseCompieInfo(expect_compile_info, actual_compile_info));
}

TEST_F(ElewiseTilingV3, ConstructTest2) {
  std::string compile_info_in = R"({ "_outs_uint1": false, "_flag_info": [true],"_base_info": {"000": [32, 2, 43680, 21840]}, "_pattern": "ElemWise"})";
  nlohmann::json op_info = nlohmann::json::parse(compile_info_in.c_str());
  ElewiseCompileInfo actual_compile_info("ElemWise", op_info);
  ElewiseCompileInfo expect_compile_info;
  expect_compile_info.has_outs_uint1 = true;
  expect_compile_info.outs_uint1 = false;
  expect_compile_info.has_flag_info = true;
  expect_compile_info.flag_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.is_const_shapes = false;
  expect_compile_info.use_special_pattern = true;
  expect_compile_info.pattern_key = 0;
  expect_compile_info.core_num = 32;
  expect_compile_info.max_dtype = 2;
  expect_compile_info.max_available_ub = 43680;
  expect_compile_info.max_available_ub_db = 21840;
  expect_compile_info.const_block_dims = -1;
  expect_compile_info.elewise_vars_size = 0;
  expect_compile_info.broadcast_pattern = false;
  ASSERT_TRUE(CompareElewiseCompieInfo(expect_compile_info, actual_compile_info));
}

TEST_F(ElewiseTilingV3, TilingTest1) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 4;
  compile_info.max_available_ub = 16384;
  compile_info.max_available_ub_db = 8192;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 0;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "8388608 8192 ");
}

TEST_F(ElewiseTilingV3, TilingTest2) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 4;
  compile_info.max_available_ub = 16384;
  compile_info.max_available_ub_db = 8192;
  compile_info.const_block_dims = 0;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling(op_info));
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "268435456 8388608 8192 ");
}

TEST_F(ElewiseTilingV3, TilingTest3) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 33, 1089}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 33, 1089}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 4;
  compile_info.max_available_ub = 32768;
  compile_info.max_available_ub_db = 16384;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "35937 1152 1152 ");
}

TEST_F(ElewiseTilingV3, TilingTest4) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 33, 1089}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 33, 1089}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 4;
  compile_info.max_available_ub = 32768;
  compile_info.max_available_ub_db = 16384;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling(op_info));
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "35937 1152 1152 ");
}

TEST_F(ElewiseTilingV3, TilingTest5) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{2, 0, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{2, 0, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = -1;
  compile_info.core_num = -1;
  compile_info.max_dtype = -1;
  compile_info.max_available_ub = -1;
  compile_info.max_available_ub_db = -1;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, TilingTest6) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{2, 0, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{2, 0, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = -1;
  compile_info.core_num = -1;
  compile_info.max_dtype = -1;
  compile_info.max_available_ub = -1;
  compile_info.max_available_ub_db = -1;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling(op_info));
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, TilingTest7) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "6 6 6 ");
}

TEST_F(ElewiseTilingV3, TilingTest8) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling(op_info));
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "6 6 6 ");
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest1) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = false;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest2) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = false;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest3) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 0;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest4) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = -1;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest5) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = -1;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest6) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = -1;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckCompileInfoTest7) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = -1;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckOpParasTest1) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckOpParasTest2) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  for (uint32_t i = 0; i < 80; i++) {
    in_shapes.push_back({3, 2});
  }
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckOpParasTest3) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckOpParasTest4) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckOpParasTest5) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}, {3, 2, 3}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, CheckOpParasTest6) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{3, 2}, {3, 1}};
  std::vector<std::vector<int64_t>> out_shapes = {{3, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 6;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 131008;
  compile_info.max_available_ub_db = 65504;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, mask_rccn_fused_mul_add_n_fail_case) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 364}, {1024, 364}, {1}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 364}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 1;
  compile_info.only_const_tiling = true;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 0;
  compile_info.core_num = 32;
  compile_info.max_dtype = 2;
  compile_info.max_available_ub = 32768;
  compile_info.max_available_ub_db = 16384;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 0;
  compile_info.broadcast_pattern = false;
  optiling::utils::OpRunInfo run_info;
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, apply_rms_prop_d_st_case) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 16}, {1, 16}, {1, 16}, {1}, {1, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 16}, {1, 16}, {1, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo compile_info;
  compile_info.has_outs_uint1 = true;
  compile_info.outs_uint1 = false;
  compile_info.has_flag_info = true;
  compile_info.flag_size = 7;
  compile_info.only_const_tiling = false;
  compile_info.is_const_shapes = false;
  compile_info.use_special_pattern = true;
  compile_info.pattern_key = 1;
  compile_info.core_num = 32;
  compile_info.max_dtype = 4;
  compile_info.max_available_ub = 9360;
  compile_info.max_available_ub_db = 4680;
  compile_info.const_block_dims = -1;
  compile_info.elewise_vars_size = 3;
  compile_info.broadcast_pattern = true;
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  Elewise elewise("ElemWise", op_paras, compile_info, run_info);
  ASSERT_TRUE(elewise.DoTiling(op_info));
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "16 16 16 ");
}