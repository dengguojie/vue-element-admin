#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/transpose_dsl.h"

using namespace optiling;
class TransposeDslTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TransposeDslTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TransposeDslTiling TearDown" << std::endl;
  }
};

enum ParamsType {
  INPUT,
  OUTPUT
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    if (i != (data.length() - sizeof(int32_t))) {
      result += ", ";
    }
  }

  return result;
}

static void AddParams(ge::OpDescPtr& op_desc, const std::vector<int64_t>& shape, const ge::DataType dtype,
                      ParamsType params_type = INPUT, ge::Format format = ge::FORMAT_ND) {
  ge::GeTensorDesc tensor;
  tensor.SetShape(ge::GeShape(shape));
  tensor.SetFormat(format);
  tensor.SetDataType(dtype);
  if (params_type == INPUT) {
    op_desc->AddInputDesc(tensor);
  } else {
    op_desc->AddOutputDesc(tensor);
  }
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_custom_unsupported) {
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [1, 0, 3, 2], "_permute": [1, 0, 3, 2], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000023": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2", "_ub_factor_3"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000122": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2000123": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2", "_ub_factor_3"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000023": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2", "_ub_factor_3"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000122": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2000123": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2", "_ub_factor_3"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000022": [], "2000023": [], "2000033": [], "2000101": [], "2000111": [], "2000122": [], "2000123": [], "2000133": [], "2000222": [], "2000223": [], "2000323": [], "2000333": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000022": [], "2000023": [], "2000033": [], "2000101": [], "2000111": [], "2000122": [], "2000123": [], "2000133": [], "2000222": [], "2000223": [], "2000323": [], "2000333": [], "3000000": []}})";
  ge::Operator op_paras = ge::Operator(this->test_info_->name());

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{};
  optiling::OpInfo c_op_info(input_shapes, ge::DT_FLOAT);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_FALSE(outer_compile_info->DoTiling(op_paras, runInfo, c_op_info));
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case1) {
  // n last transpose, fp 32, last value > 128, [0, 2, 1, 3]
  std::vector<std::vector<int64_t>> inputs {
      {32, 100, 18, 141}
  };
  std::vector<std::vector<int64_t>> outputs {
      {32, 18, 100, 141}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 2, 1, 3], "_permute": [0, 2, 1, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32, 100, 18, 141, 1, 4");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case2) {
  // n last transpose, fp 32, last value < 128, [0, 2, 1, 3]
  std::vector<std::vector<int64_t>> inputs {
      {32, 100, 18, 46}
  };
  std::vector<std::vector<int64_t>> outputs {
      {32, 18, 100, 46}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 2, 1, 3], "_permute": [0, 2, 1, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32, 100, 18, 46, 1, 6, 6");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case3) {
  // n last transpose, fp 16, [0, 2, 1, 3]
  std::vector<std::vector<int64_t>> inputs {
      {320, 29, 12, 5}
  };
  std::vector<std::vector<int64_t>> outputs {
      {320, 12, 29, 5}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 2, 1, 3], "_permute": [0, 2, 1, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "320, 29, 12, 5, 5, 2");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case4) {
  // n last transpose, fp 16, last value < 128, [0, 2, 1, 3]
  std::vector<std::vector<int64_t>> inputs {
      {32, 100, 18, 46}
  };
  std::vector<std::vector<int64_t>> outputs {
      {32, 18, 100, 46}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 2, 1, 3], "_permute": [0, 2, 1, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32, 100, 18, 46, 1, 9, 9");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case5) {
  // n last transpose, fp 16, last align, [0, 2, 1, 3]
  std::vector<std::vector<int64_t>> inputs {
      {32, 100, 18, 256}
  };
  std::vector<std::vector<int64_t>> outputs {
      {32, 18, 100, 256}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 2, 1, 3], "_permute": [0, 2, 1, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32, 100, 18, 256, 1, 16, 16");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case6) {
  // last transpose, fp16, no cross, [2, 1, 0]
  std::vector<std::vector<int64_t>> inputs {
      {100, 780, 129}
  };
  std::vector<std::vector<int64_t>> outputs {
      {129, 780, 100}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0], "_permute": [2, 1, 0], "_transpose_vars": [true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000202": [], "2000212": [], "2000222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000202": [], "2000212": [], "2000222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 28);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100, 780, 129, 7, 129, 4");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case7) {
  // last transpose, fp16, cross, [4, 3, 1, 2, 0]
  std::vector<std::vector<int64_t>> inputs {
      {28, 49, 31, 1, 25}
  };
  std::vector<std::vector<int64_t>> outputs {
      {25, 1, 49, 31, 28}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 1, 2, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0], "_permute": [2, 1, 0], "_transpose_vars": [true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000202": [], "2000212": [], "2000222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000202": [], "2000212": [], "2000222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 31);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "28, 1519, 25, 10, 5");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case8) {
  // pure copy, [0, 1]
  std::vector<std::vector<int64_t>> inputs {
      {1000, 3000}
  };
  std::vector<std::vector<int64_t>> outputs {
      {1000, 3000}
  };
  std::string compile_info = R"({"_mergeable": [0, 1], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0], "_permute": [0], "_transpose_vars": [true], "_only_const_tiling": false, "_is_const": false, "_vars": {"3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": []}, "_custom_vars": {"3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 23);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3000000, 2, 65536");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case9) {
  // last transpose, fp16, cross, [3, 2, 1, 0]
  std::vector<std::vector<int64_t>> inputs {
      {10, 200, 6, 9}
  };
  std::vector<std::vector<int64_t>> outputs {
      {9, 6, 200, 10}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [3, 2, 1, 0], "_permute": [3, 2, 1, 0], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_3"], "2000213": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_3"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_3"], "2000213": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_3"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000003": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000103": [], "2000113": [], "2000202": [], "2000212": [], "2000222": [], "2000203": [], "2000213": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000003": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000103": [], "2000113": [], "2000202": [], "2000212": [], "2000222": [], "2000203": [], "2000213": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 2);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10, 200, 6, 9, 1, 6, 102");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case10) {
  // last transpose, fp16, cross, [3, 2, 1, 0]
  std::vector<std::vector<int64_t>> inputs {
      {9, 8, 200, 2}
  };
  std::vector<std::vector<int64_t>> outputs {
      {2, 200, 8, 9}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [3, 2, 1, 0], "_permute": [3, 2, 1, 0], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_3"], "2000213": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_3"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_3"], "2000213": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_3"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000003": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000103": [], "2000113": [], "2000202": [], "2000212": [], "2000222": [], "2000203": [], "2000213": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000003": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000103": [], "2000113": [], "2000202": [], "2000212": [], "2000222": [], "2000203": [], "2000213": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "9, 8, 200, 2, 1, 200, 8");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case11) {
  // last transpose, fp16, cross, [3, 2, 1, 0]
  std::vector<std::vector<int64_t>> inputs {
      {10, 1, 12, 14}
  };
  std::vector<std::vector<int64_t>> outputs {
      {14, 12, 1, 10}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [3, 2, 1, 0], "_permute": [3, 2, 1, 0], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_3"], "2000213": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_3"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_3"], "2000213": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_3"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000003": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000103": [], "2000113": [], "2000202": [], "2000212": [], "2000222": [], "2000203": [], "2000213": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000003": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000103": [], "2000113": [], "2000202": [], "2000212": [], "2000222": [], "2000203": [], "2000213": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10, 1, 12, 14");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case12) {
  // adjust input factor, [2, 1, 0, 3], low_factor == 1
  std::vector<std::vector<int64_t>> inputs {
      {61, 333, 60, 1}
  };
  std::vector<std::vector<int64_t>> outputs {
      {60, 333, 61, 1}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0, 3], "_permute": [2, 1, 0, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 31);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "61, 333, 60, 1, 11, 1, 45");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case13) {
  // adjust output factor, [2, 1, 0, 3], high_factor == 1
  std::vector<std::vector<int64_t>> inputs {
      {60, 333, 61, 1}
  };
  std::vector<std::vector<int64_t>> outputs {
      {61, 333, 60, 1}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0, 3], "_permute": [2, 1, 0, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "60, 333, 61, 1, 21, 60, 1");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case14) {
  // adjust input factor, [2, 1, 0, 3], low_factor  != 1
  std::vector<std::vector<int64_t>> inputs {
      {61, 333, 30, 1}
  };
  std::vector<std::vector<int64_t>> outputs {
      {30, 333, 61, 1}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0, 3], "_permute": [2, 1, 0, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 28);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "61, 333, 30, 1, 6, 2, 45");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case15) {
  // adjust output factor, [2, 1, 0, 3], high_factor != 1
  std::vector<std::vector<int64_t>> inputs {
      {30, 333, 61, 1}
  };
  std::vector<std::vector<int64_t>> outputs {
      {61, 333, 30, 1}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0, 3], "_permute": [2, 1, 0, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2020002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2020101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000033": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000133": [], "2000202": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020001": [], "2020002": [], "2020101": [], "2020111": [], "2020102": [], "2020112": [], "2020202": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "30, 333, 61, 1, 11, 60, 2");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case16) {
  // [2, 1, 3, 0]
  std::vector<std::vector<int64_t>> inputs {
      {8, 660, 40, 5}
  };
  std::vector<std::vector<int64_t>> outputs {
      {40, 660, 5, 8}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 3, 0], "_permute": [3, 1, 0, 2], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000023": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000122": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000123": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2", "_ub_factor_3"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_3"], "2000023": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000122": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2000103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_3"], "2000113": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_3"], "2000123": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2", "_ub_factor_3"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_0", "_ub_factor_3"], "2000313": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_1", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000022": [], "2000003": [], "2000023": [], "2000101": [], "2000111": [], "2000122": [], "2000103": [], "2000113": [], "2000123": [], "2000222": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000022": [], "2000003": [], "2000023": [], "2000101": [], "2000111": [], "2000122": [], "2000103": [], "2000113": [], "2000123": [], "2000222": [], "2000223": [], "2000303": [], "2000313": [], "2000323": [], "2000333": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8, 660, 40, 5, 14, 27, 3");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case17) {
  // n last transpose, fp 32, last value > 128, total_size < max_ub * core_num, [0, 2, 1, 3]
  std::vector<std::vector<int64_t>> inputs {
      {32, 2, 18, 141}
  };
  std::vector<std::vector<int64_t>> outputs {
      {32, 18, 2, 141}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 2, 1, 3], "_permute": [0, 2, 1, 3], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "10000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2030000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2030001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2030002": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2030003": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2030101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2030102": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2030103": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2030202": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2030203": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_3"], "2030303": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "20000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2020000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2020011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1"], "2020012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2020022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2020111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2020112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2020212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2020222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000111": [], "2000112": [], "2000133": [], "2000212": [], "2000222": [], "2000233": [], "2000333": [], "10000": [], "2030000": [], "2030001": [], "2030002": [], "2030003": [], "2030101": [], "2030102": [], "2030103": [], "2030202": [], "2030203": [], "2030303": [], "20000": [], "2020000": [], "2020011": [], "2020012": [], "2020022": [], "2020111": [], "2020112": [], "2020212": [], "2020222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32, 2, 18, 141, 1, 18");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case18) {
  // [1, 0, 3, 2]
  std::vector<std::vector<int64_t>> inputs {
      {88, 660, 8, 22}
  };
  std::vector<std::vector<int64_t>> outputs {
      {666, 88, 22, 8}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [1, 0, 3, 2], "_permute": [1, 0, 3, 2], "_transpose_vars": [true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000023": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2", "_ub_factor_3"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000122": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2000123": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2", "_ub_factor_3"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2"], "2000023": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_2", "_ub_factor_3"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_0", "_ub_factor_3"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_1"], "2000122": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2"], "2000123": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_2", "_ub_factor_3"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_1", "_ub_factor_3"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2"], "2000223": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_2", "_ub_factor_2", "_ub_factor_3"], "2000323": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_2", "_ub_factor_3"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_block_factor_3", "_ub_factor_3"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000022": [], "2000023": [], "2000033": [], "2000101": [], "2000111": [], "2000122": [], "2000123": [], "2000133": [], "2000222": [], "2000223": [], "2000323": [], "2000333": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000022": [], "2000023": [], "2000033": [], "2000101": [], "2000111": [], "2000122": [], "2000123": [], "2000133": [], "2000222": [], "2000223": [], "2000323": [], "2000333": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "88, 660, 8, 22, 21, 1, 21");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case19) {
  // last transpose, fp16, cross, [4, 3, 1, 2, 0]
  std::vector<std::vector<int64_t>> inputs {
      {28, 49, 31, 1, 25}
  };
  std::vector<std::vector<int64_t>> outputs {
      {25, 1, 49, 31, 28}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 1, 2, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [2, 1, 0], "_permute": [2, 1, 0], "_transpose_vars": [true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0"], "2000001": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "2000002": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_2"], "2000101": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1"], "2000102": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_2"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000202": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_0", "_ub_factor_2"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_block_factor_2", "_ub_factor_2"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000202": [], "2000212": [], "2000222": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000001": [], "2000002": [], "2000101": [], "2000111": [], "2000102": [], "2000112": [], "2000202": [], "2000212": [], "2000222": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  OpInfo opInfo(inputs, dtype, std::vector<std::vector<int32_t>>());

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo, opInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 31);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "28, 1519, 25, 10, 5");
}

TEST_F(TransposeDslTiling, transpose_dsl_tiling_case20) {
  // last transpose, fp16, cross, [4, 3, 1, 2, 0]
  std::vector<std::vector<int64_t>> inputs {
      {5, 22, 41, 18, 92}
  };
  std::vector<std::vector<int64_t>> outputs {
      {5, 18, 22, 92, 41}
  };
  std::string compile_info = R"({"_mergeable": [0, 0, 0, 0, 0], "_pattern": "Transpose", "_core_num": 32, "_ub_size": 262144, "_ori_permute": [0, 3, 1, 4, 2], "_permute": [0, 2, 4, 1, 3], "_transpose_vars": [true, true, true, true, true], "_only_const_tiling": false, "_is_const": false, "_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_3"], "2000014": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_1", "_ub_factor_4"], "2000034": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_3", "_ub_factor_4"], "2000044": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_4"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_3"], "2000114": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_1", "_ub_factor_4"], "2000134": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_3", "_ub_factor_4"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_3"], "2000214": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_1", "_ub_factor_4"], "2000234": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_3", "_ub_factor_4"], "2000244": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_4"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_3", "_ub_factor_3"], "2000334": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_3", "_ub_factor_3", "_ub_factor_4"], "2000414": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_4", "_ub_factor_1", "_ub_factor_4"], "2000434": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_4", "_ub_factor_3", "_ub_factor_4"], "2000444": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_4", "_ub_factor_4"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"0": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4"], "2000000": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_0"], "2000011": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_1"], "2000012": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_1", "_ub_factor_2"], "2000022": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_2"], "2000033": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_3"], "2000014": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_1", "_ub_factor_4"], "2000034": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_3", "_ub_factor_4"], "2000044": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_0", "_ub_factor_4"], "2000111": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_1"], "2000112": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_1", "_ub_factor_2"], "2000133": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_3"], "2000114": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_1", "_ub_factor_4"], "2000134": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_1", "_ub_factor_3", "_ub_factor_4"], "2000212": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_1", "_ub_factor_2"], "2000222": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_2"], "2000233": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_3"], "2000214": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_1", "_ub_factor_4"], "2000234": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_3", "_ub_factor_4"], "2000244": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_2", "_ub_factor_4"], "2000333": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_3", "_ub_factor_3"], "2000334": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_3", "_ub_factor_3", "_ub_factor_4"], "2000414": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_4", "_ub_factor_1", "_ub_factor_4"], "2000434": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_4", "_ub_factor_3", "_ub_factor_4"], "2000444": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "_dim_4", "_block_factor_4", "_ub_factor_4"], "3000000": ["_dim_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000014": [], "2000034": [], "2000044": [], "2000111": [], "2000112": [], "2000133": [], "2000114": [], "2000134": [], "2000212": [], "2000222": [], "2000233": [], "2000214": [], "2000234": [], "2000244": [], "2000333": [], "2000334": [], "2000414": [], "2000434": [], "2000444": [], "3000000": []}, "_custom_vars": {"0": [], "2000000": [], "2000011": [], "2000012": [], "2000022": [], "2000033": [], "2000014": [], "2000034": [], "2000044": [], "2000111": [], "2000112": [], "2000133": [], "2000114": [], "2000134": [], "2000212": [], "2000222": [], "2000233": [], "2000214": [], "2000234": [], "2000244": [], "2000333": [], "2000334": [], "2000414": [], "2000434": [], "2000444": [], "3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();

  for (size_t i = 0; i < inputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    AddParams(op_desc, inputs[i], dtype, OUTPUT);
  }

  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  OpInfo opInfo(inputs, dtype, std::vector<std::vector<int32_t>>());

  nlohmann::json op_info = nlohmann::json::parse(compile_info.c_str());
  optiling::utils::OpRunInfo runInfo;
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateTransposeDslTilingHandler(this->test_info_->name(),
                                    "Transpose",
                                    op_info);
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo, opInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 30);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5, 22, 41, 18, 92, 3, 80");
}