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

#include "op_tiling/cube_tiling.h"

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
    result += " ";
  }

  return result;
}

struct Conv2DTilingTestParam {
    std::string caseName;
    std::string opType;
    std::string compileInfo;

    // input of op tiling
    std::initializer_list<int64_t> xOriginShape;
    ge::Format xOriginFormat;

    std::initializer_list<int64_t> xStorageShape;
    ge::Format xStorageFormat;

    std::initializer_list<int64_t> filterOriginShape;
    ge::Format filterOriginFormat;

    std::initializer_list<int64_t> filterStorageShape;
    ge::Format filterStorageFormat;

    std::initializer_list<int64_t> yOriginShape;
    ge::Format yOriginFormat;

    std::initializer_list<int64_t> yStorageShape;
    ge::Format yStorageFormat;

    std::vector<int64_t> stridesList;
    std::vector<int64_t> padsList;
    std::vector<int64_t> dilationsList;
    int64_t groups = 1;
    std::string dataFormat;

    // output of op tiling
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
    std::string tilingData;
};

class Conv2DTilingRuntime2: public testing::TestWithParam<Conv2DTilingTestParam> {};

TEST_P(Conv2DTilingRuntime2, general_cases) {
    Conv2DTilingTestParam param = GetParam();
    std::cout << "=========================>Run case: " << param.caseName << std::endl;

    // get functions
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(param.opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(param.opType.c_str())->tiling_parse;

    // test compile info parse
    optiling::Conv2DTilingParseInfo opInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(param.compileInfo.c_str())})
        .Outputs({&opInfo})
        .Build();
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // test op tiling
    gert::StorageShape xShape = {param.xOriginShape, param.xStorageShape};
    gert::StorageShape filterShape = {param.filterOriginShape, param.filterStorageShape};
    gert::StorageShape yShape = {param.yOriginShape, param.yStorageShape};
    std::vector<std::pair<std::string, ge::AnyValue>> attrsPairs = {
        std::make_pair("strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.stridesList)),
        std::make_pair("pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.padsList)),
        std::make_pair("dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilationsList)),
        std::make_pair("groups", ge::AnyValue::CreateFrom<int64_t>(param.groups))
        };
    auto tilingData = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&xShape, &filterShape})
        .OutputShapes({&yShape})
        .NodeAttrs(attrsPairs)
        .NodeInputTd(0, ge::DT_FLOAT16, param.xOriginFormat, param.xStorageFormat)
        .NodeInputTd(1, ge::DT_FLOAT16, param.filterOriginFormat, param.filterStorageFormat)
        .NodeOutputTd(0, ge::DT_FLOAT16, param.yOriginFormat, param.yStorageFormat)
        .CompileInfo(&opInfo)
        .TilingData(tilingData.get())
        .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    uint64_t* tilingKey = tilingContext->GetOutputPointer<uint64_t>(0);
    uint32_t* blockDim = tilingContext->GetOutputPointer<uint32_t>(1);
    std::string outputTilingData = TilingData2Str(tilingContext->GetRawTilingData());
    ASSERT_EQ(*tilingKey, param.tilingKey);
    ASSERT_EQ(*blockDim, param.blockDim);
    ASSERT_EQ(outputTilingData, param.tilingData.c_str());
}

static Conv2DTilingTestParam general_cases_params[] = {
    {
        "Conv2d_tiling_dynamic_nhw", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NHWC",
        2, 10000, "1 16 16 16 16 "
    },
    {
        "Conv2d_tiling_dynamic_batch_n", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": 1}, "tiling_range":{"10000":[1,3]}, )"\
        R"("block_dim": {"10000": 8}, "_vars": {"10000": ["batch_n"]}, "_custom_vars": {"10000": ["batch_n"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": true, "soc_version": "Ascend910A", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NCHW",
        8, 10000, "1 "
    },
    {
        "Conv2d_tiling_dynamic_None", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, )"\
        R"("block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}, "_custom_vars": {"10000": ["batch_n"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NCHW",
        1, 10000, "1 "
    },
    {
        "Conv2d_tiling_dynamic_channel", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "fmap_c1": 2, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 2, 16, 16, 16}, ge::Format::FORMAT_NC1HWC0,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 4, 16, 16, 16}, ge::Format::FORMAT_NC1HWC0,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NCHW",
        2, 10000, "1 16 16 16 16 "
    },
    {
        "Conv2d_tiling_dynamic_nhw_repo", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000":[1, 10, 10]}, "repo_range": {"10000": [1, 10, 10, 25, 10, 25]}, "cost_range": {}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 64, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NCHW",
        2, 10000, "1 16 16 16 16 "
    },
    {
        "Conv2d_tiling_fuzz_build_list_input", "Conv2D",
        R"([)"\
        R"({)"\
        R"("_pattern": "Convolution", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [16, 32, 16, 32, 16, 32]}, )"\
        R"("block_dim": {"10000": 16}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0})"\
        R"(}, )"\
        R"({"_pattern": "Convolution", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10001": [16, 32, 64, 128, 64, 128]}, )"\
        R"("block_dim": {"10001": 16}, "_vars": {"10001": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10001": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}})"\
        R"(])",
        {16, 3, 16, 16}, ge::Format::FORMAT_NCHW,
        {16, 3, 16, 16}, ge::Format::FORMAT_NCHW,
        {33, 3, 3, 5}, ge::Format::FORMAT_NCHW,
        {33, 3, 3, 5}, ge::Format::FORMAT_NCHW,
        {16, 33, 14, 12}, ge::Format::FORMAT_NCHW,
        {16, 33, 14, 12}, ge::Format::FORMAT_NCHW,
        {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW",
        16, 10000, "16 16 14 16 12 "
    },
    {
        "Conv2d_tiling_dynamic_nhwc", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NHWC",
        2, 10000, "1 16 16 16 16 "
    },
    {
        "Conv2d_vadd_fusion_tiling_dynamic_nhwc", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["_dim_0_0", "_dim_2_0", "batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NHWC",
        2, 10000, "1 256 1 16 16 16 16 "
    },
    {
        "Conv2d_vadd_fusion_tiling_dynamic_nhwc_invalid", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["dim", "batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NHWC",
        2, 10000, ""
    },
    {
        "Conv2d_vadd_fusion_tiling_dynamic_nhwc_invalid1", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, )"\
        R"("block_dim": {"10000": 2}, "_vars": {"10000": ["xyz", "batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"10000": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 32}, ge::Format::FORMAT_NHWC,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NHWC",
        2, 10000, "1 16 16 16 16 "
    },
    {
        "Conv2d_tiling_binary_case0", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 16, 16}, ge::Format::FORMAT_NCHW,
        {1, 2, 16, 16, 16}, ge::Format::FORMAT_NC1HWC0,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {16, 4, 16, 16}, ge::Format::FORMAT_FRACTAL_Z,
        {1, 16, 16, 64}, ge::Format::FORMAT_NHWC,
        {1, 4, 16, 16, 16}, ge::Format::FORMAT_NC1HWC0,
        {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, 1, "NCHW",
        2, 89, "1 32 16 16 64 3 3 1 1 1 1 16 16 1 1 1 1 0 0 1 2 1 1 288 40 2 1 2 18 40 1 18 1 1 18 "
    },
    {
        "Conv2d_tiling_binary_case1", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 4.5, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 56, 56}, ge::Format::FORMAT_NCHW,
        {1, 2, 56, 56, 16}, ge::Format::FORMAT_NC1HWC0,
        {64, 32, 1, 1}, ge::Format::FORMAT_NCHW,
        {2, 4, 16, 16}, ge::Format::FORMAT_FRACTAL_Z,
        {1, 64, 56, 56}, ge::Format::FORMAT_NCHW,
        {1, 4, 56, 56, 16}, ge::Format::FORMAT_NC1HWC0,
        {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW",
        2, 32793, "1 32 56 56 64 1 1 1 1 1 1 56 56 0 0 0 0 0 0 1 1 2 1 32 1 2 2 64 1 1 1 2 1 1 2 "
    },
    {
        "Conv2d_tiling_binary_case_cin_lessthan_16", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 15, 56, 56}, ge::Format::FORMAT_NCHW,
        {1, 1, 56, 56, 16}, ge::Format::FORMAT_NC1HWC0,
        {64, 15, 1, 1}, ge::Format::FORMAT_NCHW,
        {1, 4, 16, 16}, ge::Format::FORMAT_FRACTAL_Z,
        {1, 64, 56, 56}, ge::Format::FORMAT_NCHW,
        {1, 4, 56, 56, 16}, ge::Format::FORMAT_NC1HWC0,
        {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW",
        2, 32793, "1 16 56 56 64 1 1 1 1 1 1 56 56 0 0 0 0 0 0 1 1 2 1 16 1 4 1 64 1 1 1 1 1 1 1 "
    },
    {
        "Conv2d_tiling_binary_case_stride_2", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 56, 56}, ge::Format::FORMAT_NCHW,
        {1, 2, 56, 56, 16}, ge::Format::FORMAT_NC1HWC0,
        {64, 32, 1, 1}, ge::Format::FORMAT_NCHW,
        {2, 4, 16, 16}, ge::Format::FORMAT_FRACTAL_Z,
        {1, 64, 28, 28}, ge::Format::FORMAT_NCHW,
        {1, 4, 28, 28, 16}, ge::Format::FORMAT_NC1HWC0,
        {1, 1, 2, 2}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW",
        2, 25, "1 32 56 56 64 1 1 1 1 2 2 28 28 0 0 0 0 0 0 1 1 2 1 32 1 4 1 16 2 1 1 2 1 1 2 "
    },
    {
        "Conv2d_tiling_binary_case_dilation_2", "Conv2D",
        R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, "_custom_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
        R"("fusion_utilize": {"pre_fusion_ub_utilize": 0, "post_fusion_ub_utilize": 3, "pre_fusion_vector_utilize": 0, "post_fusion_vector_utilize": 0}, )"\
        R"("hardware_info": {"aicore_num": 2, "l2_size": 8388608, "l1_size": 1048576, "l0_a_size": 65536, "l0_b_size": 65536, )"\
        R"("l0_c_size": 262144, "ub_size": 253952, "bt_size": 0, "cube_vector_split_bool": false, "soc_version": "Ascend310", )"\
        R"("ddr_read_rate": 67, "ddr_write_rate": 64, "l2_rate": 128, "l2_read_rate": 128, "l2_write_rate": 64, "l1_to_l0_a_rate": 512, )"\
        R"("l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 64, )"\
        R"("ub_to_ddr_rate": 64, "ub_to_l1_rate": 128, "cube_bandwidth": 0, "vector_bandwidth": 0}})",
        {1, 32, 56, 56}, ge::Format::FORMAT_NCHW,
        {1, 2, 56, 56, 16}, ge::Format::FORMAT_NC1HWC0,
        {64, 32, 3, 3}, ge::Format::FORMAT_NCHW,
        {16, 4, 16, 16}, ge::Format::FORMAT_FRACTAL_Z,
        {1, 64, 52, 52}, ge::Format::FORMAT_NCHW,
        {1, 4, 52, 52, 16}, ge::Format::FORMAT_NC1HWC0,
        {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 2, 2}, 1, "NCHW",
        2, 65, "1 32 56 56 64 3 3 2 2 1 1 52 52 0 0 0 0 0 0 1 1 2 1 800 32 2 1 2 18 32 2 50 1 1 18 "
    }
};

INSTANTIATE_TEST_CASE_P(Conv2D, Conv2DTilingRuntime2, testing::ValuesIn(general_cases_params));

TEST_F(Conv2DTilingRuntime2, paddingSAME) {
    // get functions
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D"), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->tiling_parse;

    // test compile info parse
    std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
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
        .Inputs({const_cast<char*>(compileInfo.c_str())})
        .Outputs({&opInfo})
        .Build();
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // test op tiling
    gert::StorageShape xShape = {{1, 32, 16, 16}, {1, 2, 16, 16, 16}};
    gert::StorageShape filterShape = {{64, 32, 3, 3}, {16, 4, 16, 16}};
    gert::StorageShape yShape = {{1, 64, 16, 16}, {1, 4, 16, 16, 16}};
    std::vector<std::pair<std::string, ge::AnyValue>> attrsPairs = {
        std::make_pair("strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
        std::make_pair("pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})),
        std::make_pair("dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
        std::make_pair("groups", ge::AnyValue::CreateFrom<int64_t>(1)),
        std::make_pair("data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")),
        std::make_pair("offset_x", ge::AnyValue::CreateFrom<int64_t>(0)),
        std::make_pair("padding", ge::AnyValue::CreateFrom<std::string>("SAME"))
        };
    auto tilingData = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&xShape, &filterShape})
        .OutputShapes({&yShape})
        .NodeAttrs(attrsPairs)
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_FRACTAL_Z)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .CompileInfo(&opInfo)
        .TilingData(tilingData.get())
        .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    uint64_t* tilingKey = tilingContext->GetOutputPointer<uint64_t>(0);
    uint32_t* blockDim = tilingContext->GetOutputPointer<uint32_t>(1);
    std::string outputTilingData = TilingData2Str(tilingContext->GetRawTilingData());
    ASSERT_EQ(*tilingKey, 89);
    ASSERT_EQ(*blockDim, 2);
    ASSERT_EQ(outputTilingData, "1 32 16 16 64 3 3 1 1 1 1 16 16 1 1 1 1 0 0 1 2 1 1 288 40 2 1 2 18 40 1 18 1 1 18 ");
}

TEST_F(Conv2DTilingRuntime2, autoPadSAME_LOWER) {
    // get functions
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D"), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2D")->tiling_parse;

    // test compile info parse
    std::string compileInfo = R"({"_pattern": "Convolution", "push_status": 0, "tiling_type": "binary", "_vars": {"88": ["batch_n", "fmap_h", "ho", "fmap_w", "wo"]}, )"\
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
        .Inputs({const_cast<char*>(compileInfo.c_str())})
        .Outputs({&opInfo})
        .Build();
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // test op tiling
    gert::StorageShape xShape = {{1, 32, 16, 16}, {1, 2, 16, 16, 16}};
    gert::StorageShape filterShape = {{64, 32, 3, 3}, {16, 4, 16, 16}};
    gert::StorageShape yShape = {{1, 64, 16, 16}, {1, 4, 16, 16, 16}};
    std::vector<std::pair<std::string, ge::AnyValue>> attrsPairs = {
        std::make_pair("strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
        std::make_pair("pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -1, -1, -1})),
        std::make_pair("dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})),
        std::make_pair("groups", ge::AnyValue::CreateFrom<int64_t>(1)),
        std::make_pair("data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")),
        std::make_pair("offset_x", ge::AnyValue::CreateFrom<int64_t>(0)),
        std::make_pair("padding", ge::AnyValue::CreateFrom<std::string>("EXPLICIT")),
        std::make_pair("auto_pad", ge::AnyValue::CreateFrom<std::string>("SAME_LOWER"))
        };
    auto tilingData = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&xShape, &filterShape})
        .OutputShapes({&yShape})
        .NodeAttrs(attrsPairs)
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_FRACTAL_Z)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC1HWC0)
        .CompileInfo(&opInfo)
        .TilingData(tilingData.get())
        .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    uint64_t* tilingKey = tilingContext->GetOutputPointer<uint64_t>(0);
    uint32_t* blockDim = tilingContext->GetOutputPointer<uint32_t>(1);
    std::string outputTilingData = TilingData2Str(tilingContext->GetRawTilingData());
    ASSERT_EQ(*tilingKey, 89);
    ASSERT_EQ(*blockDim, 2);
    ASSERT_EQ(outputTilingData, "1 32 16 16 64 3 3 1 1 1 1 16 16 1 1 1 1 0 0 1 2 1 1 288 40 2 1 2 18 40 1 18 1 1 18 ");
}