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
#include <memory>
#include <vector>
#include "array_ops.h"
#include "nn_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling_registry.h"

namespace optiling {
struct Conv2DTilingParseInfo {
    int32_t fmapC1 = 0;
    bool correctRangeFlag = false;
    std::string tilingType = "";
    std::vector<std::string> varMap;
    std::vector<std::string> tilingKeyList;
    std::vector<std::vector<std::string>> customVarsList;
    std::vector<std::vector<int64_t>> defaultRangeList;
    std::vector<std::vector<int64_t>> tilingRangeList;
    std::vector<int32_t> blockDimList;
    std::vector<std::vector<int32_t>> repoSeedsList;
    std::vector<std::vector<int64_t>> repoRangeList;
    std::vector<std::vector<int64_t>> costRangeList;
    // hardware info
    uint32_t aicoreNum = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint32_t ddrReadRate = 0;
    uint32_t ddrWriteRate = 0;
    uint32_t l2Rate = 0;
    uint32_t l2ReadRate = 0;
    uint32_t l2WriteRate = 0;
    uint32_t l1ToL0aRate = 0;
    uint32_t l1ToL0bRate = 0;
    uint32_t l1ToUbRate = 0;
    uint32_t l0cToUbRate = 0;
    uint32_t ubToL2Rate = 0;
    uint32_t ubToDdrRate = 0;
    uint32_t ubToL1Rate = 0;
    uint32_t cubeBandwidth = 0;
    uint32_t vectorBandwidth = 0;
    bool cubeVectorSplit = false;
    std::string socVersion = "";
    // fusion utilize info
    float preFusionUbUtilize = 0;
    int64_t preFusionVectorUtilize = 0;
    float postFusionUbUtilize = 0;
    int64_t postFusionVectorUtilize = 0;
};

bool Conv2DTilingParseFunc(const std::string& opType, const nlohmann::json& opCompileInfo,
    optiling::Conv2DTilingParseInfo& opInfo);
bool Conv2DTiling(const std::string& opType, const ge::Operator& opParas,
    optiling::Conv2DTilingParseInfo& opInfo, utils::OpRunInfo& runInfo);
}

static void Conv2DTilingBenchmarkNHW(benchmark::State& state)
{
    string opType("Conv2D");
    ge::TensorDesc descX(ge::Shape({1, 32, 16, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    auto x = ge::op::Data("x");
    x.update_input_desc_x(descX);
    x.update_output_desc_y(descX);

    ge::TensorDesc descFilter(ge::Shape({64, 32, 3, 3}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    auto filter = ge::op::Data("filter").set_attr_index(1);
    filter.update_input_desc_x(descFilter);
    filter.update_output_desc_y(descFilter);

    auto conv2d = ge::op::Conv2D("Conv2D")
        .set_input_x(x)
        .set_input_filter(filter);

    ge::TensorDesc descOutput(ge::Shape({1, 64, 16, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);

    conv2d.update_input_desc_x(descX);
    conv2d.update_input_desc_filter(descFilter);
    conv2d.update_output_desc_y(descOutput);

    std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )";
    compileInfo += R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )";
    compileInfo += R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )";
    compileInfo += R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )";
    compileInfo += R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )";
    compileInfo += R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )";
    compileInfo += R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})";

    std::unique_ptr<nlohmann::json> compileInfoJsonPtr(new (std::nothrow)
        nlohmann::json(nlohmann::json::parse(compileInfo.c_str())));
    optiling::Conv2DTilingParseInfo opCompileInfo;
    optiling::Conv2DTilingParseFunc(opType, (*compileInfoJsonPtr), opCompileInfo);

    optiling::utils::OpRunInfo runInfo;
    for (auto _ : state) {
        optiling::Conv2DTiling(opType, conv2d, opCompileInfo, runInfo);
    }
}
BENCHMARK(Conv2DTilingBenchmarkNHW);