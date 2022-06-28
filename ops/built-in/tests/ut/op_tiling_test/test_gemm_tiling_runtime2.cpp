#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#define private public
#define protected public
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/cache_tiling.h"
#include "op_tiling/cube_tiling_runtime.h"

using namespace std;
using namespace ge;

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
    result += " ";
  }

  return result;
}

struct GEMMTilingTestParam {
  string case_name;
  string op_type;
  string compile_info;

  // input
  ge::Format x1_format;
  ge::Format x2_format;
  ge::Format y_format;
  bool trans_a;
  bool trans_b;
  std::initializer_list<int64_t> x1_shape;
  std::initializer_list<int64_t> x2_shape;
  std::initializer_list<int64_t> y_shape;

  bool private_attr;
  int32_t input_size;
  int32_t hidden_size;

  // output
  uint32_t block_dim;
  uint64_t tiling_key;
  string tiling_data;
};

class GEMMTilingRuntime2 : public testing::TestWithParam<GEMMTilingTestParam> {
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

namespace gert {
enum DynamicMode {
  DYNAMIC_MKN,
  DYNAMIC_MKNB
};

class GemmCompileInfo : public optiling::CubeCompileInfo {
 public:
  GemmCompileInfo() = default;
  ~GemmCompileInfo() override = default;

  bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;

  bool trans_a = false;
  bool trans_b = false;
  bool repo_seed_flag = false;
  bool repo_costmodel_flag = false;
  uint32_t workspace_num = 0;
  uint32_t ub_size = 0;
  optiling::BatchmatmulCompileParas params;
  DynamicMode dynamic_mode = DYNAMIC_MKN;
};
}

TEST_P(GEMMTilingRuntime2, general_cases) {
  GEMMTilingTestParam param = GetParam();
  std::cout << "run case " << param.case_name << std::endl;

  gert::StorageShape x1_shape = {param.x1_shape, param.x1_shape};
  gert::StorageShape x2_shape = {param.x2_shape, param.x2_shape};
  std::vector<gert::StorageShape> output_shapes(1, {param.y_shape, param.y_shape});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  gert::GemmCompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({const_cast<char *>(param.compile_info.c_str())})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  auto tiling_data = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(param.trans_a)},
                                {"adj_x2", ge::AnyValue::CreateFrom<bool>(param.trans_b)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, param.x1_format, param.x1_format)
                    .NodeInputTd(1, ge::DT_FLOAT16, param.x2_format, param.x2_format)
                    .NodeOutputTd(0, ge::DT_FLOAT16, param.y_format, param.y_format)
                    .CompileInfo(&compile_info)
                    .TilingData(tiling_data.get())
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  auto tiling_key = tiling_context->GetOutputPointer<uint64_t>(0);
  auto block_dim = tiling_context->GetOutputPointer<uint32_t>(1);
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  ASSERT_EQ(*tiling_key, param.tiling_key);
  ASSERT_EQ(*block_dim, param.block_dim);
  ASSERT_EQ(tiling_data_result, param.tiling_data);
}

static GEMMTilingTestParam general_cases_params[] = {
  {
    "GEMM_op_tiling_obj_matmul", "MatMul", R"({"_pattern": "Matmul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {2, 3}, {3, 4}, {2, 4}, false, 0, 0, 2, 10000,
    "1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_repo", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {"10114": [128, 32, 32, 32]}, "repo_range": {"10114": [126, 130, 30, 34, 30, 34, 32, 2147483647]},
        "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"10114": 32}, "correct_range_flag":null,
        "_vars":{"10114":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"10114":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"10114":[]},
        "_attr_vars":{"10114":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {32, 2048, 512}, {512, 512}, {32, 2048, 512}, false, 0, 0, 32, 10114,
    "32 128 32 32 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_formula01", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"20000000": 32}, "correct_range_flag":null,
        "_vars":{"20000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"20000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"20000000":[]},
        "_attr_vars":{"20000000":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 16, 512}, {512, 16}, {1, 16, 16}, false, 0, 0, 1, 20000000,
    "32 1 1 1 32 1 1 1 1 1 1 1 1 1 1 32 1 1 1 1 1 32 32 1 1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_formula02", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"20001000": 32}, "correct_range_flag":null,
        "_vars":{"20001000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"20001000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"20001000":[]},
        "_attr_vars":{"20001000":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 240, 768}, {768, 6000}, {1, 240, 6000}, false, 0, 0, 15, 20001000,
    "48 15 375 1 48 1 1 1 15 1 1 3 5 5 5 12 1 4 4 1 1 48 48 1 1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_formula03", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"80221000": 32}, "correct_range_flag":null,
        "_vars":{"80221000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"80221000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"80221000":[]},
        "_attr_vars":{"80221000":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 16, 2048}, {2048, 1008}, {1, 16, 1008}, false, 0, 0, 7, 80221000,
    "128 1 63 1 128 1 1 1 7 1 1 1 1 9 1 4 1 8 8 4 4 32 32 1 1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_formula04", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"70220000": 32}, "correct_range_flag":null,
        "_vars":{"70220000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"70220000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"70220000":[]},
        "_attr_vars":{"70220000":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 4096, 3136}, {3136, 1024}, {1, 4096, 1024}, false, 0, 0, 32, 70220000,
    "196 256 64 1 196 2 1 1 4 8 1 1 1 16 16 4 1 1 1 49 49 4 4 1 1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_formula05", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"62211000": 32}, "correct_range_flag":null,
        "_vars":{"62211000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"62211000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"62211000":[]},
        "_attr_vars":{"62211000":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 16000, 1024}, {1024, 512}, {1, 16000, 512}, false, 0, 0, 25, 62211000,
    "64 1000 32 1 64 5 2 1 1 25 1 1 1 16 8 4 1 4 16 4 1 16 64 4 1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_batchmatmul_formula06", "BatchMatMul", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false}, "block_dim": {"20000000": 32}, "correct_range_flag":null,
        "_vars":{"20000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"20000000":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"20000000":[]},
        "_attr_vars":{"20000000":[]}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 128, 128}, {128, 64}, {1, 128, 64}, false, 0, 0, 32, 20000000,
    "8 8 4 1 8 1 1 1 4 8 1 1 1 1 1 8 1 1 1 1 1 8 8 1 1 1 1 "
  },
  {
    "GEMM_op_tiling_arr", "MatMul", R"([{"_pattern": "Matmul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ",
        "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]},
        "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}},{"_pattern": "Matmul",
        "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {},
        "cost_range": {"10001": [1, 3, 1, 3, 4, 7]}, "block_dim": {"10001": 2}, "attrs":{"transpose_a": false,
        "transpose_b": false}}])",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {2, 3}, {3, 4}, {2, 4}, false, 0, 0, 2, 10000,
    "1 1 1 "
  },
  {
    "GEMM_op_tiling_obj_nd", "MatMul", R"({"_pattern": "Matmul", "format_a": "ND", "format_b": "ND", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 1, 3, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {2, 3}, {3, 4}, {2, 4}, false, 0, 0, 2, 10000,
    "3 2 4 "
  },
  {
    "GEMM_op_tiling_nd_aligned_pattern", "MatMul", R"({"_pattern": "Matmul", "format_a": "ND", "format_b": "ND", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 3, 2, 4, 1, 3]}, "block_dim": {"10000": 2}, "attrs":{"transpose_a": false, "transpose_b": false}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {16, 32}, {32, 16}, {16, 16}, false, 0, 0, 2, 20000,
    "32 16 16 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern", "BatchMatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
        "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
        "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mknb",
        "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {4352, 16, 64}, {4352, 64, 32}, {4352, 16, 32}, false, 0, 0, 32, 800001300,
    "64 1 2 4352 4 1 1 32 1 1 1 1 1 2 1 4 1 1 1 1 1 4 4 1 17 1 8 1 2 4 4 1 1 1 1 80 1 10240 16384 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern_02", "BatchMatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true},
        "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
        "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mknb",
        "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {4352, 64, 16}, {4352, 32, 64}, {4352, 16, 32}, false, 0, 0, 32, 800001300,
    "64 1 2 4352 4 1 1 32 1 1 1 1 1 2 1 4 1 1 1 1 1 4 4 1 17 1 8 1 2 4 4 1 1 1 1 1 80 8192 20480 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern_03", "MatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
        "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
        "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mknb",
        "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {8192, 512}, {512, 512}, {8192, 512}, false, 0, 0, 32, 622010000,
    "512 512 32 1 32 8 1 1 4 8 1 1 1 4 8 8 2 2 4 2 1 16 32 2 1 1 1 4 1 16 32 8 2 1 1 272 528 17408 8448 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern_04", "MatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, {40960, 512}, {40960, 1024}, {512, 1024}, false, 0, 0, 32, 802210000,
    "40960 32 64 1 2560 1 1 1 8 4 1 1 1 2 8 8 4 2 2 160 160 16 16 1 1 1 1 8 8 8 4 1 1 2 4 144 144 18432 9216 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern_05", "MatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mknb",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {11776, 2048}, {2048, 512}, {11776, 512}, false, 0, 0, 32, 522010100,
    "2048 736 32 1 128 4 1 1 4 8 1 1 1 1 23 2 8 2 64 32 1 4 128 32 1 1 1 23 8 4 2 1 1 1 64 80 144 29440 4608 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern_split_k", "MatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":true, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mkn",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {11776, 512}, {11776, 512}, {512, 512}, false, 0, 0, 32, 1622000100,
    "11776 32 32 736 2 1 1 4 2 4 1 1 4 8 8 2 1 23 23 1 8 184 23 1 1 1 8 8 8 4 1 1 1 46 144 144 18432 9216 "
  },
  {
    "GEMM_op_tiling_nd_nonrange_pattern_bmn_non_factor", "MatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false,"nd_flag":true, "split_k_flag":false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"correct_range_flag":null,"dynamic_mode":"dynamic_mkn",
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {2, 6640, 112}, {2, 112, 128}, {2, 6640, 128}, false, 0, 0, 32, 101010201,
    "112 415 8 7 2 1 2 4 4 1 2 1 1 32 1 2 7 7 1 1 7 7 1 1 1 1 8 2 7 7 1 8 1 1 1 1 14336 3584 "
  },
  {
    "GEMM_op_tiling_fractal_z", "MatMul", R"([{"_pattern": "Matmul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn",
      "repo_seeds": {}, "repo_range": {}, "cost_range": {}, "block_dim": {"1120000": 6},
      "attrs":{"transpose_a": false, "transpose_b": true}},{"_pattern": "Matmul", "format_a": "FRACTAL_NZ",
      "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mkn", "repo_seeds": {}, "repo_range": {}, "cost_range": {},
      "block_dim": {"1120000": 6},
      "attrs":{"transpose_a": false, "transpose_b": true}}])",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, {1, 256}, {96, 256}, {1, 96}, true, 17, 50, 6, 20000000,
    "16 1 6 16 1 1 1 6 1 1 1 1 1 1 16 1 1 1 1 1 16 16 1 1 1 1 "
  },
};

INSTANTIATE_TEST_CASE_P(Gemm, GEMMTilingRuntime2, testing::ValuesIn(general_cases_params));
