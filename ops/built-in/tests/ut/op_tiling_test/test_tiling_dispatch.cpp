#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "op_tiling/vector_tiling.h"

using namespace std;
using namespace ge;

class TilingDispatch : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "TilingDispatch SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "TilingDispatch TearDown" << std::endl;
    }
};

TEST_F(TilingDispatch, TilingDispatchElewise) {
  using namespace optiling;
  auto op = op::SoftmaxV2("TilingDispatchElemwise");
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({"_pattern": "ElemWise"})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_TRUE(op_compile_info != nullptr);
  ASSERT_TRUE(tiling_func(op, op_compile_info, runInfo));
}

TEST_F(TilingDispatch, TilingDispatchBroadcast) {
  using namespace optiling;
  auto op = op::SoftmaxV2("TilingDispatchBroadcast");
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({"_pattern": "Broadcast"})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_TRUE(op_compile_info != nullptr);
  ASSERT_TRUE(tiling_func(op, op_compile_info, runInfo));
}

TEST_F(TilingDispatch, TilingDispatchReduce) {
  using namespace optiling;
  auto op = op::SoftmaxV2("TilingDispatchCommReduce");
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({"_pattern": "CommReduce"})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_TRUE(op_compile_info != nullptr);
  ASSERT_TRUE(tiling_func(op, op_compile_info, runInfo));
}

TEST_F(TilingDispatch, TilingDispatchNorm) {
  using namespace optiling;
  auto op = op::SoftmaxV2("TilingDispatchNorm");
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 128], "_available_ub_size": {"4000": [15792, 16120, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"200400000": [32]}, "_norm_vars": {"200400000": [20000, 20001, 30000, 40000]}})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_TRUE(op_compile_info != nullptr);
  ASSERT_TRUE(tiling_func(op, op_compile_info, runInfo));
}

TEST_F(TilingDispatch, TilingDispatchTransposeDsl) {
  using namespace optiling;
  auto op = op::SoftmaxV2("TilingDispatchTransposeDsl");
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({"_pattern": "Transpose"})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_TRUE(op_compile_info != nullptr);
  ASSERT_TRUE(tiling_func(op, op_compile_info, runInfo));
}

TEST_F(TilingDispatch, TilingDispatchUnknownPattern) {
  using namespace optiling;
  auto op = op::SoftmaxV2("TilingDispatchUnknownPattern");
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({"_pattern": "TilingDispatchUnknownPattern"})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_FALSE(op_compile_info != nullptr);
  ASSERT_FALSE(tiling_func(op, op_compile_info, runInfo));
}

TEST_F(TilingDispatch, TilingDispatchUnknownOpType) {
  using namespace optiling;
  auto op = ge::Operator("TilingDispatchUnknownOpType");
  op.operator_impl_ = nullptr;
  optiling::utils::OpRunInfo runInfo;
  std::string op_name = "AutoTiling";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  optiling::OpTilingFuncInfo& op_func_info = iter->second;
  ASSERT_TRUE(op_func_info.IsFunctionV4());
  const OpTilingFuncV4& tiling_func = op_func_info.GetOpTilingFuncV4();
  const OpParseFuncV4& parse_func = op_func_info.GetOpParseFuncV4();
  ge::AscendString compileInfo(R"({"_pattern": "TilingDispatchUnknownOpType"})");
  std::shared_ptr<CompileInfoBase> op_compile_info = parse_func(op, compileInfo);
  ASSERT_FALSE(op_compile_info != nullptr);
  ASSERT_FALSE(tiling_func(op, op_compile_info, runInfo));
}