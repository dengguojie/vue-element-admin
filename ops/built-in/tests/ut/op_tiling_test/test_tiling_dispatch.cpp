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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({"_pattern": "ElemWise"})");
  void* op_compile_info = parse_func(op, compileInfo);
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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({"_pattern": "Broadcast"})");
  void* op_compile_info = parse_func(op, compileInfo);
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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({"_pattern": "CommReduce"})");
  void* op_compile_info = parse_func(op, compileInfo);
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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({ "_fuse_axis": false, "_ori_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 1, 21448, 21496, 16216, 128], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_reduce_shape_known": true, "_const_shape_post": false})");
  void* op_compile_info = parse_func(op, compileInfo);
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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({"_pattern": "Transpose"})");
  void* op_compile_info = parse_func(op, compileInfo);
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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({"_pattern": "TilingDispatchUnknownPattern"})");
  void* op_compile_info = parse_func(op, compileInfo);
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
  ASSERT_TRUE(op_func_info.IsFunctionV3());
  const OpTilingFuncV3& tiling_func = op_func_info.GetOpTilingFuncV3();
  const OpParseFuncV3& parse_func = op_func_info.GetOpParseFuncV3();
  ge::AscendString compileInfo(R"({"_pattern": "TilingDispatchUnknownOpType"})");
  void* op_compile_info = parse_func(op, compileInfo);
  ASSERT_FALSE(op_compile_info != nullptr);
  ASSERT_FALSE(tiling_func(op, op_compile_info, runInfo));
}