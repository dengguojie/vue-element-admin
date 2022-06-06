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
#include "op_tiling/tiling_handler.h"

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

static bool CompareRequiredInfo(const ElewiseCompileInfo& expect_compile_info,
                                const ElewiseCompileInfo& real_compile_info) {
  if (expect_compile_info.classify_inputs_num != real_compile_info.classify_inputs_num) {
    std::cout << "The classify_inputs_num is wrong!" << std::endl;
    return false;
  }
  if (expect_compile_info.flag_info_size != real_compile_info.flag_info_size) {
    std::cout << "The flag_info_size is wrong!" << std::endl;
    return false;
  }
  if (expect_compile_info.only_const_tiling != real_compile_info.only_const_tiling) {
    std::cout << "The only_const_tiling is wrong!" << std::endl;
    return false;
  }
  if (expect_compile_info.ub_factor_align != real_compile_info.ub_factor_align) {
    std::cout << "The ub_factor_align is wrong!" << std::endl;
    return false;
  }
  return true;
}

static bool CompareOptionalInfo(const ElewiseCompileInfo& expect_compile_info,
                                const ElewiseCompileInfo& real_compile_info) {
  if (expect_compile_info.classify_const_mode != real_compile_info.classify_const_mode) {
    std::cout << "The classify_const_mode is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.support_broadcast != real_compile_info.support_broadcast) {
    std::cout << "The support_broadcast is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.absorbable_broadcast != real_compile_info.absorbable_broadcast) {
    std::cout << "The absorbable_broadcast is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.const_block_dims.first != real_compile_info.const_block_dims.first) {
    std::cout << "The const_block_dims.first is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.const_block_dims.second != real_compile_info.const_block_dims.second) {
    std::cout << "The const_block_dims.second vector is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.base_info.first != real_compile_info.base_info.first) {
    std::cout << "The base_info.first is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.base_info.first == real_compile_info.base_info.first) {
    if (!CompareMap(expect_compile_info.base_info.second, real_compile_info.base_info.second)) {
      return false;
    }
  }
  if (expect_compile_info.elewise_vars.first != real_compile_info.elewise_vars.first) {
    std::cout << "The elewise_vars.first is wrong" << std::endl;
    return false;
  }
  if (expect_compile_info.elewise_vars.first == real_compile_info.elewise_vars.first) {
    if (!CompareMap(expect_compile_info.elewise_vars.second, real_compile_info.elewise_vars.second)) {
      return false;
    }
  }
  return true;
}

static bool CompareElewiseCompileInfo(const ElewiseCompileInfo& expect_compile_info,
                                      const ElewiseCompileInfo& real_compile_info) {
  bool ret = true;
  ret = ret && CompareRequiredInfo(expect_compile_info, real_compile_info);
  ret = ret && CompareOptionalInfo(expect_compile_info, real_compile_info);
  return ret;
}

// Test CreateElewiseTilingHandler
TEST_F(ElewiseTilingV3, parse_compile_info_test) {
  std::string compile_info_in = R"({ "_classify_inputs_num": 1,
                                     "_ub_factor_align": 128,
                                     "_pattern": "ElemWise",
                                     "_flag_info": [false, false, false, true, false, false],
                                     "_base_info": {"100": [32, 4, 16384, 8192]},
                                     "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ],
                                                        "210010000": [ 10000, 20000, 30000 ] },
                                     "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ],
                                                "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
  nlohmann::json op_info = nlohmann::json::parse(compile_info_in.c_str());
  auto parsed_ptr =
    std::static_pointer_cast<ElewiseTilingHandler>(CreateElewiseTilingHandler("elewise_handler1", "ElemWise", op_info));
  ASSERT_TRUE(parsed_ptr);
}

TEST_F(ElewiseTilingV3, running_tiling_info_compare) {
  std::string compile_info_in = R"({ "_classify_inputs_num": 1,
                                     "_ub_factor_align": 128,
                                     "_pattern": "ElemWise",
                                     "_flag_info": [false, false, false, true, false, false],
                                     "_base_info": {"100": [32, 4, 16384, 8192]},
                                     "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ],
                                                        "210010000": [ 10000, 20000, 30000 ] },
                                     "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ],
                                                "210010000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
  nlohmann::json op_info = nlohmann::json::parse(compile_info_in.c_str());
  ElewiseCompileInfo actual_compile_info("ElemWise", op_info);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  ASSERT_TRUE(CompareElewiseCompileInfo(expect_compile_info, actual_compile_info));
}

TEST_F(ElewiseTilingV3, compile_tiling_info_compare) {
  std::string compile_info_in = R"({ "_classify_inputs_num": 1,
                                     "_ub_factor_align": 128,
                                     "_flag_info": [true],
                                     "_base_info": {"000": [32, 2, 43680, 21840]},
                                     "_pattern": "ElemWise"})";
  nlohmann::json op_info = nlohmann::json::parse(compile_info_in.c_str());
  ElewiseCompileInfo actual_compile_info("ElemWise", op_info);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
  expect_compile_info.classify_inputs_num = 1;
  expect_compile_info.flag_info_size = 1;
  expect_compile_info.only_const_tiling = true;
  expect_compile_info.ub_factor_align = 128;
  // optional compile info
  expect_compile_info.classify_const_mode = false;
  expect_compile_info.support_broadcast = false;
  expect_compile_info.absorbable_broadcast = false;
  expect_compile_info.const_block_dims.first = false;
  expect_compile_info.base_info.first = true;
  expect_compile_info.base_info.second = {{"000", {32, 2, 43680, 21840}}};
  expect_compile_info.elewise_vars.first = false;
  ASSERT_TRUE(CompareElewiseCompileInfo(expect_compile_info, actual_compile_info));
}

TEST_F(ElewiseTilingV3, elewise_set_attr_case1) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  optiling::utils::OpRunInfo run_info;
  op_paras.SetAttr("alpha", 123);
  std::string compile_info_in = R"({"_ub_factor_align": 128,
                                    "_classify_inputs_num": 1,
                                    "_pattern": "ElemWise",
                                    "_flag_info": [false, false, false, true, false, false],
                                    "_base_info": {"100": [32, 4, 16384, 8192]},
                                    "_elewise_vars": {"210000000": [ 10000, 20000, 30000 ],
                                                      "210010000": [ 10000, 20000, 30000 ]},
                                    "_vars": {"210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ],
                                              "210010000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ]},
                                    "_var_attr_mode": 1,
                                    "_var_attrs": {"210000000": [{"length":1, "name": "alpha", "index": 0,
                                                                  "type": "int32", "src_type": "int32"}],
                                                   "210010000": [{"length":1, "name": "alpha", "index": 0,
                                                                  "type": "int32", "src_type": "int32"}]}
                                   })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info =
    CreateElewiseTilingHandler(this->test_info_->name(), "autotiling", nlohmann::json::parse(compile_info_in));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, run_info));

  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "268435456 8388608 8192 123 ");
}

TEST_F(ElewiseTilingV3, rl_hit_second_cpt_no_blocktiling) {
  // rl hit -1_-1 cpt: second target -> no block tiling
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{65540}, {65540}};
  std::vector<std::vector<int64_t>> out_shapes = {{65540}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);

  optiling::utils::OpRunInfo runInfo;

  std::string compileInfo = R"({"_ub_factor_align": 128, "_pattern": "ElemWise",
                                "_classify_inputs_num": 2,
                                "_flag_info": [false, false, true, true, true, false],
                                "_base_info": {}, "_elewise_vars": {},
                                "_bank_info": {"-1_-1":
                                [[[[[0]], [-1], [4],[[67584, 2147483647]]], ["dim_0_0"], [[0, [0], "_block_factor_0", 32, 32], [0, [0], 9400], []], 9223372038164746484, [10000, 30000, 20000]],
                                 [[[[0]], [-1], [4], [[256,65540]]], ["dim_0_0"], [[0, [0],"",32,16], [0, [0], 4152], []], 9223372040192759146, [10000, 30000]]]}
                                })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateElewiseTilingHandler(this->test_info_->name(),
                               "autotiling",
                               nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(to_string(runInfo.GetTilingKey()), "9223372040192759146");
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "65540 2056 ");
}

TEST_F(ElewiseTilingV3, rl_hit_second_cpt_blocktiling) {
  // rl hit -1_-1 cpt: second target -> block tiling
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{655400}, {655400}};
  std::vector<std::vector<int64_t>> out_shapes = {{655400}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);

  optiling::utils::OpRunInfo runInfo;

  std::string compileInfo = R"({"_ub_factor_align": 128, "_pattern": "ElemWise",
                                "_classify_inputs_num": 2,
                                "_flag_info": [false, false, true, true, true, false],
                                "_base_info": {}, "_elewise_vars": {},
                                "_bank_info": {"-1_-1":
                                [[[[[0]], [-1], [4],[[67584, 2147483647]]], ["dim_0_0"], [[0, [0], "_block_factor_0", 32, 32], [0, [0], 9400], []], 9223372038164746484, [10000, 30000, 20000]],
                                 [[[[0]], [-1], [4], [[256,65540]]], ["dim_0_0"], [[0, [0],"",32,16], [0, [0], 4152], []], 9223372040192759146, [10000, 30000]]]}
                                })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateElewiseTilingHandler(this->test_info_->name(),
                               "autotiling",
                               nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(to_string(runInfo.GetTilingKey()), "9223372038164746484");
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "655400 9400 3 ");
}

TEST_F(ElewiseTilingV3, empty_mode_none_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{2, 0, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{2, 0, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  AutoTilingOp auto_tiling_op("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(run_info.GetTilingKey(), 2147483647);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, empty_mode_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{2, 0, 2}};
  std::vector<std::vector<int64_t>> out_shapes = {{2, 0, 2}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  AutoTilingOp auto_tiling_op("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(run_info.GetTilingKey(), 2147483647);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, common_none_custom_tiling_multicore) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  AutoTilingOp auto_tiling_op("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 210010000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "268435456 8388608 8192 ");
}

TEST_F(ElewiseTilingV3, common_custom_tiling_multicore) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{128, 128, 128, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{128, 128, 128, 128}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 210010000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "268435456 8388608 8192 ");
}

TEST_F(ElewiseTilingV3, common_none_custom_tiling_single_core) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{256},{256}};
  std::vector<std::vector<int64_t>> out_shapes = {{256}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(run_info.GetTilingKey(), 210000000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "256 256 256 ");
}

TEST_F(ElewiseTilingV3, common_custom_tiling_single_core) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{256},{256}};
  std::vector<std::vector<int64_t>> out_shapes = {{256}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(run_info.GetTilingKey(), 210000000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "256 256 256 ");
}

TEST_F(ElewiseTilingV3, only_const_tiling_none_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "1 0 128 0 128 0 ");
}

TEST_F(ElewiseTilingV3, only_const_tiling_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "1 0 128 0 128 0 ");
}

TEST_F(ElewiseTilingV3, const_pattern_none_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(run_info.GetTilingKey(), 100000001);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, const_pattern_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(run_info.GetTilingKey(), 100000001);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, common_relu_fp32) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{275614, 11}};
  std::vector<std::vector<int64_t>> out_shapes = {{275614, 11}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 210010000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "3031754 94848 15872 ");
}

TEST_F(ElewiseTilingV3, broadcast_scalar_custom_tiling_pattern) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{22, 16384, 4}, {1}};
  std::vector<std::vector<int64_t>> out_shapes = {{22, 16384, 4}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 223010000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "1441792 45056 15104 ");
}

TEST_F(ElewiseTilingV3, scalar_broadcast_none_custom_tiling_pattern) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1, 1}, {16, 7, 4, 35}};
  std::vector<std::vector<int64_t>> out_shapes = {{16, 7, 4, 35}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_INT32, ge::DT_INT32};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT32};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 31);
  EXPECT_EQ(run_info.GetTilingKey(), 232000000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "15680 512 512 ");
}

TEST_F(ElewiseTilingV3, broadcast_addcmul_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{275614, 11}, {1, 1}, {275614, 11}, {275614, 11}};
  std::vector<std::vector<int64_t>> out_shapes = {{275614, 11}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 220010000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "3031754 3031754 3031754 94848 11904 ");
}

TEST_F(ElewiseTilingV3, apply_adam_d_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33},
                                                 {1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33},
                                                 {1024, 256, 33}, {1024, 256, 33}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 256, 33}, {1024, 256, 33}, {1024, 256, 33}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                         ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                         ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 210010000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "8650752 270336 8064 ");
}

TEST_F(ElewiseTilingV3, fuse_mul_add_n_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 364}, {1024, 364}, {1}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 364}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 210000000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "372736 11648 11648 ");
}

TEST_F(ElewiseTilingV3, ub_bound_limit_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{16384, 1115}};
  std::vector<std::vector<int64_t>> out_shapes = {{16384, 1115}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT16};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, elewise_var_key_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{275614, 11}};
  std::vector<std::vector<int64_t>> out_shapes = {{275614, 11}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, low_shape_same_none_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 256}, {1, 123}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 256}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, low_shape_same_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 256}, {1, 123}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 256}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, diff_lens_higher_shape_all_one_none_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 256}, {256}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 256}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, diff_lens_higher_shape_all_one_custom_check) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1024, 256}, {256}};
  std::vector<std::vector<int64_t>> out_shapes = {{1024, 256}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_FALSE(elewise.DoTiling());
}

TEST_F(ElewiseTilingV3, apply_rms_prop_d_st_fail_case) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 16}, {1, 16}, {1, 16}, {1}, {1, 16}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 16}, {1, 16}, {1, 16}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 1);
  EXPECT_EQ(run_info.GetTilingKey(), 210000000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "16 16 16 ");
}

TEST_F(ElewiseTilingV3, dynamic_add_const_elewise_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, dynamic_add_const_elewise_custom_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}, {1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
  std::vector<ge::DataType> out_dtypes = {ge::DT_FLOAT};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND, ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
    // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::OpInfo op_info(in_shapes, in_dtypes[0]);
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, OpInfoImplGetter::GetOpInfoImpl(&op_info).get());
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}

TEST_F(ElewiseTilingV3, dynamic_cast_s32_to_s64_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_INT32};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT64};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  optiling::AutoTilingOp oldTlingop("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&oldTlingop, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 4);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "1024 256 256 ");
}

TEST_F(ElewiseTilingV3, dynamic_cast_s64_to_s32_tiling) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{1, 1024}};
  std::vector<std::vector<int64_t>> out_shapes = {{1, 1024}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_INT64};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT32};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  AutoTilingOp auto_tiling_op("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 8);
  EXPECT_EQ(run_info.GetTilingKey(), 210000000);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "1024 128 128 ");
}

TEST_F(ElewiseTilingV3, const_vcmp_support_b64_case) {
  // Construct op_paras
  std::vector<std::vector<int64_t>> in_shapes = {{11323, 128}, {11323, 128}};
  std::vector<std::vector<int64_t>> out_shapes = {{11323, 128}};
  std::vector<ge::DataType> in_dtypes = {ge::DT_INT64,ge::DT_INT64};
  std::vector<ge::DataType> out_dtypes = {ge::DT_INT8};
  std::vector<ge::Format> in_formats = {ge::FORMAT_ND};
  std::vector<ge::Format> out_formats = {ge::FORMAT_ND};
  const ge::Operator op_paras = ConstructOpParas(in_shapes, out_shapes, in_dtypes,
                                                 out_dtypes, in_formats, out_formats);
  ElewiseCompileInfo expect_compile_info;
  // required compile info
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
  optiling::utils::OpRunInfo run_info;
  AutoTilingOp auto_tiling_op("ElemWise", &op_paras, &expect_compile_info, &run_info);
  v3::Elewise<AutoTilingOp> elewise(&auto_tiling_op, nullptr);
  ASSERT_TRUE(elewise.DoTiling());
  EXPECT_EQ(run_info.GetBlockDim(), 32);
  EXPECT_EQ(run_info.GetTilingKey(), 100000001);
  EXPECT_EQ(to_string(run_info.GetAllTilingData()), "");
}
