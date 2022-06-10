#define protected public

#include <vector>
#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

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
}

static void Conv2DTilingRuntime2BenchmarkNHW(benchmark::State& state) {
    std::string opType("Conv2D");
    gert::StorageShape xShape = {{1, 16, 16, 32}, {1, 16, 16, 32}};
    gert::StorageShape filterShape = {{64, 32, 3, 3}, {64, 32, 3, 3}};
    gert::StorageShape yShape = {{1, 16, 16, 64}, {1, 16, 16, 64}};

    const char* compile_str = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})";

    optiling::Conv2DTilingParseInfo opInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(compile_str)})
        .Outputs({&opInfo})
        .Build();
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>());

    auto tilingData = gert::TilingData::CreateCap(sizeof(opInfo));
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&xShape, &filterShape})
        .OutputShapes({&yShape})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_NHWC)
        .CompileInfo(&opInfo)
        .TilingData(tilingData.get())
        .Build();
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    for (auto _ : state) {
        tilingFunc(holder.GetContext<gert::TilingContext>());
    }
}
BENCHMARK(Conv2DTilingRuntime2BenchmarkNHW);

static void Conv2DTilingRuntime2BenchmarkBinary1(benchmark::State& state) {
    std::string opType("Conv2D");
    gert::StorageShape xShape = {{1, 32, 16, 16}, {1, 2, 16, 16, 16}};
    gert::StorageShape filterShape = {{64, 32, 3, 3}, {16, 4, 16, 16}};
    gert::StorageShape yShape = {{1, 64, 16, 16}, {1, 4, 16, 16, 16}};

    std::string compile_str = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})";

    optiling::Conv2DTilingParseInfo opInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(compile_str.c_str())})
        .Outputs({&opInfo})
        .Build();
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>());

    auto tilingData = gert::TilingData::CreateCap(sizeof(opInfo));
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&xShape, &filterShape})
        .OutputShapes({&yShape})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_FRACTAL_Z)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .CompileInfo(&opInfo)
        .TilingData(tilingData.get())
        .Build();
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    for (auto _ : state) {
        tilingFunc(holder.GetContext<gert::TilingContext>());
    }
}
BENCHMARK(Conv2DTilingRuntime2BenchmarkBinary1);

static void Conv2DTilingRuntime2BenchmarkBinaryAutoPad(benchmark::State& state) {
    std::string opType("Conv2D");
    gert::StorageShape xShape = {{1, 32, 16, 16}, {1, 2, 16, 16, 16}};
    gert::StorageShape filterShape = {{64, 32, 3, 3}, {16, 4, 16, 16}};
    gert::StorageShape yShape = {{1, 64, 16, 16}, {1, 4, 16, 16, 16}};

    std::string compile_str = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})";

    optiling::Conv2DTilingParseInfo opInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(compile_str.c_str())})
        .Outputs({&opInfo})
        .Build();
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>());

    auto tilingData = gert::TilingData::CreateCap(sizeof(opInfo));
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&xShape, &filterShape})
        .OutputShapes({&yShape})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_FRACTAL_Z)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
            {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
            {"padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")},
            {"auto_pad", ge::AnyValue::CreateFrom<std::string>("SAME_LOWER")}
            })
        .CompileInfo(&opInfo)
        .TilingData(tilingData.get())
        .Build();
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    for (auto _ : state) {
        tilingFunc(holder.GetContext<gert::TilingContext>());
    }
}
BENCHMARK(Conv2DTilingRuntime2BenchmarkBinaryAutoPad);
