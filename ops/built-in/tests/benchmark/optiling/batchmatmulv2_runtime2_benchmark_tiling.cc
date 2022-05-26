/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define protected public

#include <benchmark/benchmark.h>

#include <nlohmann/json.hpp>

#include "cache_tiling.h"
#include "cube_tiling_runtime.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace gert {

enum DynamicMode { DYNAMIC_MKN, DYNAMIC_MKNB };

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
}  // namespace gert

using gert::InferShapeContext;
using gert::TilingContext;
using gert::GemmCompileInfo;
using gert::TilingData;
using nlohmann::json;

static void BatchMatMulV2Tiling_repo_size_10_runtime2(benchmark::State &state) {
  std::string op_type("BatchMatMulV2");
  gert::StorageShape x1_shape = {{32, 2048, 512}, {32, 32, 128, 16, 16}};
  gert::StorageShape x2_shape = {{512, 512}, {32, 32, 16, 16}};
  std::vector<gert::StorageShape> output_shapes(1, {{32, 2048, 512}, {32, 32, 128, 16, 16}});
  std::vector<gert::StorageShape *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  // repo range size is 10
  const char *json_str =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode": "dynamic_mknb", "repo_seeds": {"20114": [128, 32, 32, 32], "10100": [126, 31, 30, 1451662655], "10101": [127, 34, 33, 440662882], "10102": [127, 30, 32, 1904033346], "10103": [126, 34, 32, 712500835], "10104": [129, 34, 33, 2140048072], "10105": [126, 30, 31, 752420050], "10106": [130, 32, 33, 672697576], "10107": [128, 33, 31, 815859467], "10108": [128, 30, 32, 1436355865], "10109": [126, 30, 32, 1999276457]}, "repo_range": {"20114": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10100": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10101": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10102": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10103": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10104": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10105": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10106": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10107": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10108": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10109": [126, 130, 30, 34, 30, 34, 32, 2147483647]}, "attrs": {"transpose_a": false, "transpose_b": false}, "block_dim": {"10114": 32, "20114": 12, "10100": 29, "10101": 4, "10102": 26, "10103": 26, "10104": 28, "10105": 17, "10106": 14, "10107": 8, "10108": 6, "10109": 21}, "correct_range_flag": null, "_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_custom_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars": {"10114": []}, "_attr_vars": {"10114": []}})";

  gert::GemmCompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                           .KernelIONum(1, 1)
                           .Inputs({const_cast<char *>(json_str)})
                           .Outputs({&compile_info})
                           .Build();

  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling_parse;
  tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());

  // tiling data
  auto param = TilingData::CreateCap(2048);

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", ge::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .CompileInfo(&compile_info)
                    .TilingData(param.get())
                    .Build();

  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling;

  for (auto _ : state) {
    tiling_func(holder.GetContext<TilingContext>());
  }

  // auto tiling_key = holder.GetContext<TilingContext>()->GetOutputPointer<uint64_t>(0);
  // auto block_dim = holder.GetContext<TilingContext>()->GetOutputPointer<uint32_t>(1);
  // std::cout << "tiling_key: " << *tiling_key << std::endl;
  // std::cout << "block_dim: " << *block_dim << std::endl;
}
BENCHMARK(BatchMatMulV2Tiling_repo_size_10_runtime2);

static void BatchMatMulV2Tiling_repo_size_100_runtime2(benchmark::State &state) {
  std::string op_type("BatchMatMulV2");
  gert::StorageShape x1_shape = {{32, 2048, 512}, {32, 32, 128, 16, 16}};
  gert::StorageShape x2_shape = {{512, 512}, {32, 32, 16, 16}};
  std::vector<gert::StorageShape> output_shapes(1, {{32, 2048, 512}, {32, 32, 128, 16, 16}});

  std::vector<gert::StorageShape *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  // repo range size is 100 and all range match
  const char *json_str =
      R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode": "dynamic_mknb", "repo_seeds": {"20114": [128, 32, 32, 32], "10100": [127, 31, 31, 2063572729], "10101": [129, 33, 34, 1144288869], "10102": [130, 34, 33, 542458225], "10103": [128, 31, 33, 1845816735], "10104": [127, 32, 33, 1981846807], "10105": [130, 31, 30, 746798279], "10106": [127, 34, 34, 1744087121], "10107": [129, 31, 30, 1490833690], "10108": [129, 33, 34, 523931990], "10109": [128, 32, 33, 1386735851], "10110": [127, 30, 34, 1720623460], "10111": [130, 31, 34, 706584842], "10112": [126, 30, 30, 1742127526], "10113": [126, 33, 32, 149430593], "10114": [126, 34, 31, 2071435011], "10115": [126, 31, 33, 1169836599], "10116": [127, 34, 30, 882197734], "10117": [129, 34, 33, 602280550], "10118": [127, 33, 32, 1705556525], "10119": [127, 33, 30, 193847877], "10120": [128, 32, 30, 202210317], "10121": [126, 33, 30, 308067334], "10122": [130, 32, 33, 1656906679], "10123": [126, 31, 32, 686187631], "10124": [127, 30, 31, 1336327661], "10125": [126, 34, 33, 100259308], "10126": [127, 34, 30, 1843386269], "10127": [126, 32, 32, 2058851880], "10128": [127, 33, 31, 401625322], "10129": [129, 34, 32, 1621542372], "10130": [127, 34, 32, 710900013], "10131": [129, 32, 32, 623660082], "10132": [130, 30, 34, 357176923], "10133": [129, 32, 33, 1271905230], "10134": [130, 34, 33, 2001943059], "10135": [129, 30, 33, 441258622], "10136": [130, 34, 32, 143961714], "10137": [129, 31, 33, 728080679], "10138": [128, 32, 34, 611056784], "10139": [126, 34, 33, 799108847], "10140": [126, 30, 34, 241396480], "10141": [128, 30, 32, 1827602906], "10142": [127, 32, 31, 1821569058], "10143": [128, 33, 33, 847878945], "10144": [127, 32, 31, 1762025930], "10145": [129, 33, 31, 1882863521], "10146": [126, 34, 31, 748926908], "10147": [130, 30, 32, 1344792142], "10148": [128, 34, 30, 1887307841], "10149": [127, 30, 34, 143554792], "10150": [127, 34, 30, 1620380575], "10151": [129, 33, 30, 292341265], "10152": [128, 31, 34, 380562631], "10153": [130, 33, 34, 1811801600], "10154": [128, 30, 31, 797260499], "10155": [126, 30, 32, 970881597], "10156": [129, 32, 31, 126016409], "10157": [126, 32, 34, 429398585], "10158": [128, 33, 30, 262756364], "10159": [128, 32, 30, 705723264], "10160": [126, 33, 34, 1415960869], "10161": [128, 33, 34, 1619543393], "10162": [128, 32, 31, 1479174869], "10163": [130, 31, 32, 1513843528], "10164": [130, 31, 32, 1969498896], "10165": [129, 30, 33, 638336027], "10166": [129, 33, 31, 950728230], "10167": [129, 33, 31, 1920908173], "10168": [127, 32, 33, 1382291852], "10169": [126, 31, 32, 399806303], "10170": [126, 32, 31, 1586758758], "10171": [126, 33, 34, 537957027], "10172": [129, 33, 30, 680040042], "10173": [126, 30, 30, 565090238], "10174": [128, 30, 33, 778322092], "10175": [128, 34, 31, 2077763846], "10176": [130, 30, 33, 1318157113], "10177": [129, 34, 33, 392698721], "10178": [126, 34, 33, 1608404457], "10179": [126, 34, 34, 2035976165], "10180": [127, 34, 30, 1735794447], "10181": [127, 33, 32, 417612363], "10182": [130, 33, 31, 857453308], "10183": [129, 34, 33, 1984058242], "10184": [129, 30, 31, 1172728599], "10185": [129, 34, 32, 188008111], "10186": [128, 30, 30, 1619867194], "10187": [126, 33, 30, 1454015047], "10188": [127, 30, 30, 1158946212], "10189": [129, 33, 32, 653602690], "10190": [127, 31, 30, 2101302303], "10191": [126, 33, 31, 739948345], "10192": [128, 30, 33, 2129483113], "10193": [128, 34, 32, 1017629926], "10194": [126, 31, 30, 1761474765], "10195": [127, 33, 32, 1232751135], "10196": [126, 34, 30, 2047823352], "10197": [127, 33, 34, 810095617], "10198": [130, 32, 33, 9024439], "10199": [129, 31, 33, 714915630]}, "repo_range": {"20114": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10100": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10101": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10102": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10103": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10104": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10105": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10106": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10107": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10108": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10109": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10110": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10111": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10112": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10113": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10114": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10115": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10116": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10117": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10118": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10119": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10120": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10121": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10122": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10123": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10124": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10125": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10126": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10127": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10128": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10129": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10130": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10131": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10132": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10133": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10134": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10135": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10136": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10137": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10138": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10139": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10140": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10141": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10142": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10143": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10144": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10145": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10146": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10147": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10148": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10149": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10150": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10151": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10152": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10153": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10154": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10155": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10156": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10157": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10158": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10159": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10160": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10161": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10162": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10163": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10164": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10165": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10166": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10167": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10168": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10169": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10170": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10171": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10172": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10173": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10174": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10175": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10176": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10177": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10178": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10179": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10180": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10181": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10182": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10183": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10184": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10185": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10186": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10187": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10188": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10189": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10190": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10191": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10192": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10193": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10194": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10195": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10196": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10197": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10198": [126, 130, 30, 34, 30, 34, 32, 2147483647], "10199": [126, 130, 30, 34, 30, 34, 32, 2147483647]}, "attrs": {"transpose_a": false, "transpose_b": false}, "block_dim": {"10114": 8, "20114": 14, "10100": 27, "10101": 25, "10102": 19, "10103": 19, "10104": 12, "10105": 2, "10106": 31, "10107": 28, "10108": 1, "10109": 11, "10110": 21, "10111": 10, "10112": 7, "10113": 2, "10115": 11, "10116": 4, "10117": 25, "10118": 14, "10119": 28, "10120": 11, "10121": 3, "10122": 8, "10123": 12, "10124": 26, "10125": 5, "10126": 8, "10127": 23, "10128": 26, "10129": 9, "10130": 6, "10131": 31, "10132": 32, "10133": 1, "10134": 28, "10135": 7, "10136": 9, "10137": 7, "10138": 21, "10139": 19, "10140": 1, "10141": 6, "10142": 26, "10143": 20, "10144": 6, "10145": 22, "10146": 6, "10147": 23, "10148": 27, "10149": 20, "10150": 19, "10151": 1, "10152": 9, "10153": 10, "10154": 27, "10155": 31, "10156": 1, "10157": 4, "10158": 25, "10159": 30, "10160": 14, "10161": 27, "10162": 7, "10163": 25, "10164": 15, "10165": 18, "10166": 21, "10167": 28, "10168": 6, "10169": 20, "10170": 29, "10171": 19, "10172": 5, "10173": 1, "10174": 1, "10175": 20, "10176": 11, "10177": 32, "10178": 32, "10179": 1, "10180": 10, "10181": 31, "10182": 9, "10183": 3, "10184": 24, "10185": 17, "10186": 19, "10187": 8, "10188": 4, "10189": 2, "10190": 27, "10191": 11, "10192": 5, "10193": 20, "10194": 28, "10195": 18, "10196": 4, "10197": 1, "10198": 26, "10199": 4}, "correct_range_flag": null, "_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_custom_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars": {"10114": []}, "_attr_vars": {"10114": []}})";

  gert::GemmCompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                           .KernelIONum(1, 1)
                           .Inputs({const_cast<char *>(json_str)})
                           .Outputs({&compile_info})
                           .Build();

  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
  tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());

  // tiling data
  auto param = TilingData::CreateCap(2048);

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", ge::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                    .CompileInfo(&compile_info)
                    .TilingData(param.get())
                    .Build();

  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling;

  for (auto _ : state) {
    tiling_func(holder.GetContext<TilingContext>());
  }

  // auto tiling_key = holder.GetContext<TilingContext>()->GetOutputPointer<uint64_t>(0);
  // auto block_dim = holder.GetContext<TilingContext>()->GetOutputPointer<uint32_t>(1);
  // std::cout << "tiling_key: " << *tiling_key << std::endl;
  // std::cout << "block_dim: " << *block_dim << std::endl;
}
BENCHMARK(BatchMatMulV2Tiling_repo_size_100_runtime2);

// static void BatchMatMulV2Tiling_cost_size_100_runtime2(benchmark::State &state) {
//   std::string op_type("BatchMatMulV2");
//   gert::StorageShape x1_shape = {{32, 2048, 512}, {32, 32, 128, 16, 16}};
//   gert::StorageShape x2_shape = {{512, 512}, {32, 32, 16, 16}};
//   std::vector<gert::StorageShape> output_shapes(1, {{32, 2048, 512}, {32, 32, 128, 16, 16}});

//   std::vector<gert::StorageShape *> output_shapes_ref(1);
//   for (size_t i = 0; i < output_shapes.size(); ++i) {
//     output_shapes_ref[i] = &output_shapes[i];
//   }

//   // repo range size is 0 and 100 cost range
//   const char *json_str =
//       R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode": "dynamic_mknb", "cost_range": {"10100": [106, 453, 20, 160, 266, 320, 49, 458], "10101": [52, 222, 67, 376, 112, 329, 28, 131], "10102": [103, 422, 51, 400, 41, 459, 170, 500], "10103": [454, 500, 120, 486, 102, 487, 300, 394], "10104": [19, 380, 366, 464, 161, 446, 9, 197], "10105": [50, 453, 40, 430, 259, 414, 18, 380], "10106": [133, 439, 17, 433, 195, 456, 152, 378], "10107": [5, 474, 169, 338, 200, 296, 38, 487], "10108": [321, 373, 145, 357, 214, 264, 283, 479], "10109": [25, 428, 130, 457, 56, 489, 105, 331], "10110": [24, 472, 71, 418, 414, 493, 23, 452], "10111": [218, 483, 179, 420, 90, 339, 1, 100], "10112": [95, 445, 66, 397, 352, 497, 129, 428], "10113": [4, 170, 223, 470, 160, 468, 26, 372], "10114": [100, 204, 95, 463, 150, 325, 52, 284], "10115": [275, 415, 163, 498, 273, 447, 32, 470], "10116": [92, 397, 92, 436, 65, 427, 155, 467], "10117": [60, 387, 25, 469, 101, 408, 26, 439], "10118": [9, 488, 180, 455, 63, 486, 128, 393], "10119": [154, 442, 10, 278, 303, 493, 39, 462], "10120": [139, 298, 150, 474, 259, 475, 39, 465], "10121": [41, 270, 231, 336, 187, 444, 97, 430], "10122": [197, 354, 28, 217, 204, 283, 82, 432], "10123": [20, 87, 25, 343, 60, 500, 175, 278], "10124": [139, 271, 211, 448, 107, 232, 290, 496], "10125": [124, 465, 236, 452, 137, 376, 385, 487], "10126": [224, 457, 62, 452, 274, 368, 8, 438], "10127": [349, 442, 265, 395, 316, 498, 340, 460], "10128": [283, 433, 113, 384, 275, 484, 4, 443], "10129": [391, 496, 131, 479, 231, 467, 290, 388], "10130": [176, 215, 167, 369, 449, 498, 250, 462], "10131": [59, 193, 107, 392, 26, 393, 29, 490], "10132": [38, 327, 61, 332, 32, 241, 222, 487], "10133": [60, 483, 188, 410, 71, 213, 126, 375], "10134": [88, 237, 35, 377, 70, 488, 70, 276], "10135": [128, 319, 8, 489, 90, 440, 27, 364], "10136": [134, 482, 230, 299, 137, 489, 198, 483], "10137": [337, 392, 259, 412, 216, 372, 202, 424], "10138": [136, 381, 331, 372, 1, 491, 25, 228], "10139": [261, 418, 98, 456, 52, 182, 7, 155], "10140": [2, 114, 346, 494, 20, 438, 35, 210], "10141": [236, 406, 46, 158, 394, 472, 166, 442], "10142": [75, 279, 39, 220, 5, 297, 47, 486], "10143": [244, 404, 14, 254, 102, 416, 62, 103], "10144": [264, 410, 1, 429, 15, 233, 286, 414], "10145": [390, 475, 36, 59, 63, 389, 477, 495], "10146": [249, 469, 346, 490, 119, 336, 22, 499], "10147": [200, 303, 47, 299, 3, 434, 7, 194], "10148": [47, 498, 72, 468, 40, 99, 10, 191], "10149": [142, 325, 263, 417, 144, 471, 151, 376], "10150": [355, 499, 81, 264, 276, 340, 21, 297], "10151": [56, 477, 3, 223, 2, 359, 23, 350], "10152": [224, 463, 358, 475, 176, 342, 118, 367], "10153": [293, 496, 166, 322, 248, 377, 148, 436], "10154": [206, 485, 1, 113, 303, 489, 460, 476], "10155": [28, 427, 392, 428, 110, 375, 6, 461], "10156": [47, 242, 63, 499, 204, 463, 48, 459], "10157": [78, 471, 298, 390, 263, 490, 125, 445], "10158": [7, 152, 19, 467, 31, 395, 61, 414], "10159": [48, 458, 35, 206, 46, 211, 310, 458], "10160": [121, 245, 12, 231, 230, 388, 20, 224], "10161": [72, 274, 94, 444, 10, 240, 76, 205], "10162": [377, 474, 120, 219, 11, 175, 226, 482], "10163": [50, 148, 34, 199, 460, 493, 354, 439], "10164": [309, 387, 104, 294, 309, 461, 60, 489], "10165": [28, 403, 128, 221, 18, 426, 72, 160], "10166": [343, 455, 117, 244, 41, 342, 19, 191], "10167": [5, 211, 71, 391, 142, 329, 44, 442], "10168": [57, 345, 223, 439, 106, 306, 449, 496], "10169": [221, 342, 18, 396, 210, 418, 64, 452], "10170": [121, 474, 109, 353, 125, 487, 16, 356], "10171": [91, 269, 9, 340, 71, 487, 72, 421], "10172": [119, 444, 56, 195, 16, 218, 4, 331], "10173": [50, 373, 94, 347, 328, 432, 281, 463], "10174": [41, 260, 1, 449, 95, 414, 118, 375], "10175": [5, 159, 93, 487, 115, 381, 6, 262], "10176": [22, 288, 388, 461, 29, 466, 83, 450], "10177": [194, 287, 52, 492, 353, 495, 1, 65], "10178": [393, 444, 47, 486, 192, 311, 76, 479], "10179": [220, 402, 219, 276, 59, 500, 358, 451], "10180": [19, 303, 143, 267, 17, 360, 101, 367], "10181": [61, 146, 39, 450, 83, 263, 136, 251], "10182": [342, 476, 341, 414, 316, 500, 160, 411], "10183": [39, 284, 37, 267, 247, 475, 200, 440], "10184": [116, 493, 36, 294, 30, 370, 379, 463], "10185": [73, 340, 84, 175, 102, 254, 100, 492], "10186": [56, 476, 170, 382, 113, 499, 4, 215], "10187": [65, 142, 3, 139, 100, 273, 48, 301], "10188": [292, 412, 8, 454, 72, 242, 178, 498], "10189": [102, 455, 116, 355, 221, 442, 285, 337], "10190": [96, 207, 20, 404, 10, 342, 307, 480], "10191": [117, 489, 166, 407, 39, 434, 65, 447], "10192": [34, 226, 216, 433, 30, 46, 360, 484], "10193": [15, 384, 202, 234, 93, 422, 21, 150], "10194": [368, 478, 264, 440, 97, 469, 291, 451], "10195": [81, 389, 105, 265, 6, 308, 62, 412], "10196": [130, 455, 90, 497, 42, 371, 109, 360], "10197": [36, 420, 122, 300, 79, 255, 89, 373], "10198": [5, 216, 76, 211, 185, 389, 39, 341], "10199": [96, 422, 33, 238, 273, 464, 45, 493]}, "attrs": {"transpose_a": false, "transpose_b": false}, "block_dim": {"10114": 16, "10100": 32, "10101": 9, "10102": 13, "10103": 11, "10104": 14, "10105": 13, "10106": 6, "10107": 21, "10108": 19, "10109": 21, "10110": 25, "10111": 28, "10112": 9, "10113": 17, "10115": 15, "10116": 10, "10117": 19, "10118": 1, "10119": 13, "10120": 26, "10121": 25, "10122": 19, "10123": 27, "10124": 29, "10125": 17, "10126": 15, "10127": 30, "10128": 32, "10129": 11, "10130": 22, "10131": 19, "10132": 20, "10133": 21, "10134": 4, "10135": 16, "10136": 24, "10137": 27, "10138": 24, "10139": 16, "10140": 15, "10141": 4, "10142": 11, "10143": 11, "10144": 11, "10145": 1, "10146": 26, "10147": 24, "10148": 18, "10149": 10, "10150": 26, "10151": 30, "10152": 29, "10153": 9, "10154": 29, "10155": 10, "10156": 30, "10157": 17, "10158": 13, "10159": 3, "10160": 26, "10161": 23, "10162": 9, "10163": 9, "10164": 26, "10165": 31, "10166": 19, "10167": 15, "10168": 10, "10169": 5, "10170": 30, "10171": 5, "10172": 27, "10173": 4, "10174": 28, "10175": 18, "10176": 20, "10177": 17, "10178": 8, "10179": 20, "10180": 24, "10181": 3, "10182": 18, "10183": 25, "10184": 9, "10185": 12, "10186": 26, "10187": 9, "10188": 29, "10189": 28, "10190": 1, "10191": 14, "10192": 6, "10193": 8, "10194": 14, "10195": 21, "10196": 31, "10197": 2, "10198": 4, "10199": 2}, "correct_range_flag": null, "_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_custom_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars": {"10114": []}, "_attr_vars": {"10114": []}})";

//   gert::GemmCompileInfo compile_info;
//   auto kernel_holder = gert::KernelRunContextFaker()
//                            .KernelIONum(1, 1)
//                            .Inputs({const_cast<char *>(json_str)})
//                            .Outputs({&compile_info})
//                            .Build();

//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
//   tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());

//   // tiling data
//   auto param = TilingData::CreateCap(2048);

//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(2, 1)
//                     .IrInstanceNum({1, 1})
//                     .InputShapes({&x1_shape, &x2_shape})
//                     .OutputShapes(output_shapes_ref)
//                     .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(false)},
//                                 {"adj_x2", ge::AnyValue::CreateFrom<bool>(false)}})
//                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
//                     .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
//                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
//                     .CompileInfo(&compile_info)
//                     .TilingData(param.get())
//                     .Build();

//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling;

//   for (auto _ : state) {
//     tiling_func(holder.GetContext<TilingContext>());
//   }

//   // auto tiling_key = holder.GetContext<TilingContext>()->GetOutputPointer<uint64_t>(0);
//   // auto block_dim = holder.GetContext<TilingContext>()->GetOutputPointer<uint32_t>(1);
//   // std::cout << "tiling_key: " << *tiling_key << std::endl;
//   // std::cout << "block_dim: " << *block_dim << std::endl;
// }
// BENCHMARK(BatchMatMulV2Tiling_cost_size_100_runtime2);

// static void BatchMatMulV2Tiling_binary_mode_runtime2(benchmark::State &state) {
//   std::string op_type("BatchMatMulV2");
//   gert::StorageShape x1_shape = {{32, 2048, 512}, {32, 32, 128, 16, 16}};
//   gert::StorageShape x2_shape = {{512, 512}, {32, 32, 16, 16}};
//   std::vector<gert::StorageShape> output_shapes(1, {{32, 2048, 512}, {32, 32, 128, 16, 16}});

//   std::vector<gert::StorageShape *> output_shapes_ref(1);
//   for (size_t i = 0; i < output_shapes.size(); ++i) {
//     output_shapes_ref[i] = &output_shapes[i];
//   }

//   // no repo range and cost range
//   const char *json_str =
//       R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode": "dynamic_mknb",
// "attrs": {"transpose_a": false, "transpose_b": false},
// "block_dim": {"10195": 21, "10196": 31, "10197": 2, "10198": 4, "10199": 2},
// "correct_range_flag": null, "_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core",
// "batch_dim", "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
// "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
// "_custom_vars": {"10114": ["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim", "n_dim",
// "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor", "kal1_factor",
// "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars": {"10114": []}, "_attr_vars": {"10114": []}})";

//   gert::GemmCompileInfo compile_info;
//   auto kernel_holder = gert::KernelRunContextFaker()
//                            .KernelIONum(1, 1)
//                            .Inputs({const_cast<char *>(json_str)})
//                            .Outputs({&compile_info})
//                            .Build();

//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
//   tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());

//   // tiling data
//   auto param = TilingData::CreateCap(2048);

//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(2, 1)
//                     .IrInstanceNum({1, 1})
//                     .InputShapes({&x1_shape, &x2_shape})
//                     .OutputShapes(output_shapes_ref)
//                     .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(false)},
//                                 {"adj_x2", ge::AnyValue::CreateFrom<bool>(false)}})
//                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
//                     .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
//                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
//                     .CompileInfo(&compile_info)
//                     .TilingData(param.get())
//                     .Build();

//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling;

//   for (auto _ : state) {
//     tiling_func(holder.GetContext<TilingContext>());
//   }

//   // auto tiling_key = holder.GetContext<TilingContext>()->GetOutputPointer<uint64_t>(0);
//   // auto block_dim = holder.GetContext<TilingContext>()->GetOutputPointer<uint32_t>(1);
//   // std::cout << "tiling_key: " << *tiling_key << std::endl;
//   // std::cout << "block_dim: " << *block_dim << std::endl;
// }
// BENCHMARK(BatchMatMulV2Tiling_binary_mode_runtime2);
