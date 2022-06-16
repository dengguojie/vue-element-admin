#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "matrix_calculation_ops.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#define private public
#define protected public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/cache_tiling.h"
#include "op_tiling/cube_tiling_runtime.h"

using namespace std;
using namespace ge;
using namespace op;

struct Conv2DBpFilterTilingTestParam {
  string case_name;
  string compile_info;

  std::initializer_list<int64_t> fmap_ori_shape;
  std::initializer_list<int64_t> fmap_shape;
  std::initializer_list<int64_t> filter_shape;
  std::initializer_list<int64_t> out_backprop_ori_shape;
  std::initializer_list<int64_t> out_backprop_shape;

  ge::Format fmap_ori_format;
  ge::Format fmap_format;
  ge::Format filter_format;
  ge::Format out_backprop_ori_format;
  ge::Format out_backprop_format;

  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  int64_t groups;
  std::string data_format;
  std::string padding;

  bool parse_result;
  bool tiling_result;

  // output
  uint32_t block_dim;
  uint64_t tiling_key;
  std::string tiling_data;
};

class Conv2DBackpropFilterTilingRunTime2 : public testing::TestWithParam<Conv2DBpFilterTilingTestParam> {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropFilterTilingRunTime2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropFilterTilingRunTime2 TearDown" << std::endl;
  }
};

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
    result += " ";
  }

  return result;
}

TEST_P(Conv2DBackpropFilterTilingRunTime2, general_cases) {
  Conv2DBpFilterTilingTestParam param = GetParam();
  std::cout << "run case " << param.case_name << std::endl;

  gert::StorageShape filter_sizes = {param.filter_shape, param.filter_shape};
  gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
  gert::StorageShape fmap_shape = {param.fmap_ori_shape, param.fmap_shape};
  std::vector<gert::StorageShape> output_shapes(1, {param.filter_shape, param.filter_shape});
  std::vector<void *> output_shapes_ref(1);

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  optiling::Conv2DBackPropCompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({const_cast<char *>(param.compile_info.c_str())})
                    .Outputs({&compile_info})
                    .Build();

  std::string op_type("Conv2DBackpropFilter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
  if (param.parse_result) {
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  } else {
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
    return;
  }

  auto tiling_data = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&fmap_shape, &filter_sizes, &out_backprop_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                {"groups", ge::AnyValue::CreateFrom<int64_t>(param.groups)},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>(param.data_format)},
                                {"padding", ge::AnyValue::CreateFrom<std::string>(param.padding)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, param.fmap_ori_format, param.fmap_format)
                    .NodeInputTd(1, ge::DT_FLOAT16, param.filter_format, param.filter_format)
                    .NodeInputTd(2, ge::DT_FLOAT16, param.out_backprop_ori_format, param.out_backprop_format)
                    .NodeOutputTd(0, ge::DT_FLOAT16, param.filter_format, param.filter_format)
                    .CompileInfo(&compile_info)
                    .TilingData(tiling_data.get())
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  if (param.tiling_result) {
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  } else {
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    return;
  }
  auto tiling_key = tiling_context->GetOutputPointer<uint64_t>(0);
  auto block_dim = tiling_context->GetOutputPointer<uint32_t>(1);
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  ASSERT_EQ(*tiling_key, param.tiling_key);
  ASSERT_EQ(*block_dim, param.block_dim);
  ASSERT_EQ(tiling_data_result, param.tiling_data);
}

static Conv2DBpFilterTilingTestParam general_cases_params[] = {
  {"Conv2d_bp_filter_tiling_dynamic_nhw", R"({"_pattern": "Conv2d_backprop_filter", "push_status": 1, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": [8, 52, 635]}, "repo_range": {"10000": [8, 8, 52, 52, 635, 635]}, "cost_range": {}, "block_dim": {"10000": 16}, "correct_range_flag": false, "_vars": {"10000": ["batch", "fmap_h", "dedy_h", "fmap_w", "dedy_w"]}})",
    {8, 5, 52, 635}, {8, 1, 52, 635, 16}, {257, 5, 1, 1}, {8, 257, 13, 159}, {8, 17, 13, 159, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 16, 10000, "8 52 13 635 159 "
  },

  {"Conv2d_bp_filter_tiling_dynamic_n", R"({"_pattern": "Conv2d_backprop_filter", "push_status": 1, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "tiling_range": {"10000": [1, 7]}, "block_dim": {"10000": 16}, "correct_range_flag": true, "_vars": {"10000": ["batch"]}})",
    {8, 5, 52, 635}, {8, 1, 52, 635, 16}, {257, 5, 1, 1}, {8, 257, 13, 159}, {8, 17, 13, 159, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, false, 16, 10000, "8 52 13 635 159 "
  },

  {"Conv2d_bp_filter_tiling_dynamic_compile_info_empty", R"({})",
    {8, 5, 52, 635}, {8, 1, 52, 635, 16}, {257, 5, 1, 1}, {8, 257, 13, 159}, {8, 17, 13, 159, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    false, false, 16, 10000, "8 52 13 635 159 "
  },

  {"Conv2d_bp_filter_tiling_dynamic_compile_info_not_have_vars", R"({"_pattern": "Conv2d_backprop_filter", "push_status": 1, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": [8, 52, 635]}, "repo_range": {"10000": [8, 8, 52, 52, 635, 635]}, "cost_range": {}, "block_dim": {"10000": 16}, "correct_range_flag": false})",
    {8, 5, 52, 635}, {8, 1, 52, 635, 16}, {257, 5, 1, 1}, {8, 257, 13, 159}, {8, 17, 13, 159, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    false, false, 16, 10000, "8 52 13 635 159 "
  },

  {"Conv2d_bp_filter_tiling_fuzz_build_list_input", R"([{"_pattern": "Conv2d_backprop_filter", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [1, 16, 50, 53, 630, 640]}, "block_dim": {"0": 16}, "_vars": {"0": ["batch", "fmap_h", "dedy_h", "fmap_w", "dedy_w"]}},{"_pattern": "Conv2d_backprop_filter", "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [16, 32, 64, 128, 64, 128]}, "block_dim": {"1": 16}, "_vars": {"1": ["batch", "fmap_h", "dedy_h", "fmap_w", "dedy_w"]}}])",
    {8, 5, 52, 635}, {8, 1, 52, 635, 16}, {257, 5, 1, 1}, {8, 257, 13, 159}, {8, 17, 13, 159, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 16, 0, "8 52 13 635 159 "
  },

  {"Conv2d_bp_filter_tiling_binary_mode_normal", R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 16, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":1,"stride_w":1}})",
    {8, 5, 56, 56}, {8, 1, 56, 56, 16}, {257, 5, 3, 3}, {8, 257, 54, 54}, {8, 17, 54, 54, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 12, 14100625, "8 5 56 56 257 54 54 3 3 1 17 1 1 0 0 0 0 1 1 1 1 4 3 2 1 1 1 9 1 1 1 1 17 1 1 1 61 61 61 61 1 19712 1 1 1 22 20 9 17 61 2 "
  },

  {"Conv2d_bp_filter_tiling_binary_mode_pads_upadate", R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 16, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":1,"stride_w":1}})",
    {8, 5, 56, 56}, {8, 1, 56, 56, 16}, {257, 5, 3, 3}, {8, 257, 56, 56}, {8, 17, 56, 56, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 1, 1}, {-1, -1, -1, -1}, {1, 1, 1, 1}, 1, "NCHW", "SAME",
    true, true, 28, 23866250, "8 5 56 56 257 56 56 3 3 1 17 1 1 1 1 1 1 1 1 1 1 4 7 2 1 1 1 3 3 1 1 1 17 2 1 1 14 14 28 28 1 8960 1 1 1 10 8 9 17 28 2 "
  },

  {"Conv2d_bp_filter_tiling_binary_mode_kl0_max_l1", R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 32, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":37,"stride_w":34}})",
    {11, 5, 150, 628}, {11, 1, 150, 628, 16}, {4, 5, 1, 13}, {11, 4, 5, 19}, {11, 1, 5, 19, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 37, 34}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "SAME",
    true, true, 33, 12584375, "11 5 150 628 4 5 19 1 13 1 1 37 34 0 0 0 0 1 1 1 1 11 3 1 1 1 1 1 13 1 1 1 1 1 1 2 2 1 2 1 2 381824 1 1 1 2 2 13 1 2 0 "
  },

  {"Conv2d_bp_filter_tiling_binary_mode_kernel_nhwc", R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 32, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":37,"stride_w":34}})",
    {11, 5, 150, 628}, {11, 1, 150, 628, 16}, {4, 5, 1, 13}, {11, 4, 5, 19}, {11, 1, 5, 19, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 37, 34}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "SAME",
    true, true, 33, 12584375, "11 5 150 628 4 5 19 1 13 1 1 37 34 0 0 0 0 1 1 1 1 11 3 1 1 1 1 1 13 1 1 1 1 1 1 2 2 1 2 1 2 381824 1 1 1 2 2 13 1 2 0 "
  },

  {"Conv2d_bp_filter_tiling_binary_mode_filter_hwcn", R"({"_pattern": "Conv2d_backprop_filter", "block_dim": {"CORE_NUM": 32}, "tiling_type": "binary", "max_core_num": 32, "attrs": {"dilation_h":1,"dilation_w":1,"groups":1,"padd":0,"padl":0,"padr":0,"padu":0,"stride_h":37,"stride_w":34}})",
    {11, 5, 150, 628}, {11, 1, 150, 628, 16}, {4, 5, 1, 13}, {11, 4, 5, 19}, {11, 1, 5, 19, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
    {1, 1, 37, 34}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "SAME",
    true, true, 33, 12584375, "11 5 150 628 4 5 19 1 13 1 1 37 34 0 0 0 0 1 1 1 1 11 3 1 1 1 1 1 13 1 1 1 1 1 1 2 2 1 2 1 2 381824 1 1 1 2 2 13 1 2 0 "
  }
};

INSTANTIATE_TEST_CASE_P(Dw, Conv2DBackpropFilterTilingRunTime2, testing::ValuesIn(general_cases_params));