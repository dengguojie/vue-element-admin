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

struct Conv2DBpInputTilingTestParam {
  string case_name;
  string compile_info;

  std::initializer_list<int64_t> input_size;
  std::initializer_list<int64_t> filter_shape;
  std::initializer_list<int64_t> filter_origin_shape;
  std::initializer_list<int64_t> out_backprop_shape;
  std::initializer_list<int64_t> out_backprop_origin_shape;
  std::initializer_list<int64_t> y_shape;
  std::initializer_list<int64_t> y_origin_shape;

  ge::Format input_size_format;
  ge::Format filter_format;
  ge::Format out_backprop_ori_format;
  ge::Format out_backprop_format;
  ge::Format y_format;
  ge::Format y_ori_format;

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

class Conv2DBackpropInputTilingRunTime2 : public testing::TestWithParam<Conv2DBpInputTilingTestParam> {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBpInputTilingTestParam SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBpInputTilingTestParam TearDown" << std::endl;
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

static std::unique_ptr<uint8_t[]> InitListIntAttr(const std::vector<int32_t> &list_int_attr) {
  auto attr_ptr = gert::ContinuousVector::Create<int32_t>(list_int_attr.size());
  auto attr = reinterpret_cast<gert::ContinuousVector *>(attr_ptr.get());
  size_t copy_size = list_int_attr.size() * sizeof(int32_t);
  (void)memcpy_s(attr->MutableData(), copy_size, list_int_attr.data(), copy_size);
  return attr_ptr;
}

  TEST_P(Conv2DBackpropInputTilingRunTime2, general_cases) {
  Conv2DBpInputTilingTestParam param = GetParam();
  std::cout << "run case " << param.case_name << std::endl;

  gert::StorageShape input_size = {param.input_size, param.input_size};
  gert::StorageShape filter_shape = {param.filter_origin_shape, param.filter_shape};
  gert::StorageShape out_backprop_shape = {param.out_backprop_origin_shape, param.out_backprop_shape};
  std::vector<gert::StorageShape> output_shapes(1, {param.y_origin_shape, param.y_shape});
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

  std::string op_type("Conv2DBackpropInput");
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
                    .InputShapes({&input_size, &filter_shape, &out_backprop_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                {"groups", ge::AnyValue::CreateFrom<int64_t>(param.groups)},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>(param.data_format)},
                                {"padding", ge::AnyValue::CreateFrom<std::string>(param.padding)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, param.input_size_format, param.input_size_format)
                    .NodeInputTd(1, ge::DT_FLOAT16, param.filter_format, ge::FORMAT_FRACTAL_Z)
                    .NodeInputTd(2, ge::DT_FLOAT16, param.out_backprop_ori_format, param.out_backprop_format)
                    .NodeOutputTd(0, ge::DT_FLOAT16, param.y_ori_format, param.y_format)
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

static Conv2DBpInputTilingTestParam general_cases_params[] = {
  {"Conv2d_bp_input_tiling_dynamic_nhw", R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})",
    {1, 32, 16, 16}, {64, 32, 3, 3}, {64, 32, 3, 3}, {1, 64, 16, 16}, {1, 64, 16, 16}, {1, 32, 16, 16}, {1, 32, 16, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 2, 10000, "1 16 16 16 16 "
  },

  {"Conv2d_bp_input_tiling_compile_info_empty", R"({})",
    {1, 32, 16, 16}, {64, 32, 3, 3}, {64, 32, 3, 3}, {1, 64, 16, 16}, {1, 64, 16, 16}, {1, 32, 16, 16}, {1, 32, 16, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    false, false, 2, 10000, "1 16 16 16 16 "
  },

  {"Conv2d_bp_input_tiling_compile_info_not_have_vars", R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}})",
    {1, 32, 16, 16}, {64, 32, 3, 3}, {64, 32, 3, 3}, {1, 64, 16, 16}, {1, 64, 16, 16}, {1, 32, 16, 16}, {1, 32, 16, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    false, false, 2, 10000, "1 16 16 16 16 "
  },

  {"Conv2d_bp_input_tiling_no_repo_seeds", R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "cost_range": {"10000": [1, 10, 10, 25, 10, 25]}, "block_dim": {"10000": 2}, "_vars": {"10000": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}})",
    {1, 32, 16, 16}, {64, 32, 3, 3}, {64, 32, 3, 3}, {1, 64, 16, 16}, {1, 64, 16, 16}, {1, 32, 16, 16}, {1, 32, 16, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 2, 10000, "1 16 16 16 16 "
  },

  {"Conv2d_bp_input_dynamic_None", R"({"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "default_tiling", "default_range": {"10000": [1, 2147483647, 16, 16, 16, 16]}, "block_dim": {"10000": 1}, "_vars": {"10000": ["batch_n"]}})",
    {1, 32, 16, 16}, {64, 32, 3, 3}, {64, 32, 3, 3}, {1, 64, 16, 16}, {1, 64, 16, 16}, {1, 32, 16, 16}, {1, 32, 16, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 1, 10000, "1 "
  },

  {"Conv2d_bp_input_fuzz_build_list_input", R"([{"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"0": [1, 10, 10, 25, 10, 25]}, "block_dim": {"0": 2}, "_vars": {"0": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}, {"_pattern": "Conv2d_backprop_input", "push_status": 0, "tiling_type": "dynamic_tiling", "repo_seeds": {}, "repo_range": {}, "cost_range": {"1": [10, 100, 15, 30, 15, 30]}, "block_dim": {"1": 2}, "_vars": {"1": ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]}}])",
    {1, 32, 16, 16}, {64, 32, 3, 3}, {64, 32, 3, 3}, {1, 64, 16, 16}, {1, 64, 16, 16}, {1, 32, 16, 16}, {1, 32, 16, 16},
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 2, 0, "1 16 16 16 16 "
  },

  {"Conv2d_bp_input_binary_stride_large_one", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 1, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {16, 4, 16, 16}, {64, 64, 2, 2}, {8, 4, 13, 10, 16}, {8, 64, 13, 10}, {8, 64, 26, 20}, {8, 64, 26, 20},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 2, 2}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 32, 21210103, "16 4 8 4 13 10 64 4 26 20 2 2 1 4 1 0 0 0 0 2 2 0 0 0 0 1 1 0 0 8 1 4 1 1 0 4 10 10 9 1 4 4 4 1 1 12800 16384 12800 "
  },

  {"Conv2d_bp_input_binary_stride_large_one_m2_n2_large_x0", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 1, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {288, 32, 16, 16}, {512, 512, 3, 3}, {128, 32, 7, 7, 16}, {128, 512, 7, 7}, {128, 512, 14, 14}, {128, 512, 14, 14},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 2, 2}, {0, 1, 0, 1}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 32, 12211023, "288 32 128 32 7 7 512 32 14 14 3 3 1 32 1 0 1 0 1 2 2 0 0 0 0 2 2 0 0 32 1 1 4 0 1 32 10 7 13 1 8 4 18 4 1 106496 147456 71680 "
  },

  {"Conv2d_bp_input_binary_stride_equal_one", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 2, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {9, 1, 16, 16}, {16, 16, 3, 3}, {4, 16, 13, 13}, {4, 16, 15, 15}, {4, 16, 15, 15}, {4, 16, 15, 15},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 1, 11210002, "4 16 13 13 9 1 16 1 15 15 3 3 1 1 1 0 0 0 0 1 1 0 0 0 0 2 2 2 2 1 1 1 4 0 0 1 13 13 15 1 1 3 3 1 1 2816 2304 2816 "
  },

  {"Conv2d_bp_input_binary_pads_equal_neg_one", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 2, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {144, 16, 16, 16}, {256, 256, 3, 3}, {16, 256, 12, 20}, {16, 256, 12, 20}, {16, 256, 12, 20}, {16, 256, 12, 20},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {-1, -1, -1, -1}, {1, 1, 1, 1}, 1, "NCHW", "SAME",
    true, true, 32, 11210002, "16 256 12 20 144 16 256 16 12 20 3 3 1 16 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 16 2 1 1 0 0 16 6 20 15 1 8 4 36 1 1 61440 294912 32768 "
  },

  {"Conv2d_bp_input_binary_dilations_invalid", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}})",
    {4}, {144, 16, 16, 16}, {256, 256, 3, 3}, {16, 256, 12, 20}, {16, 256, 12, 20}, {16, 256, 12, 20}, {16, 256, 12, 20},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 1, 1}, {-1, -1, -1, -1}, {2, 2, 2, 2}, 1, "NCHW", "",
    true, false, 32, 21210101, ""
  },

  {"Conv2d_bp_input_binary_no_overlap_condition_4", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 1, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {1, 1, 16, 16}, {16, 1, 1, 1}, {1, 1, 2, 2, 16}, {1, 16, 2, 2}, {1, 1, 4, 3}, {1, 1, 4, 3},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 2, 2}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 1, 11210003, "1 1 1 1 2 2 1 1 4 3 1 1 1 1 1 0 0 0 0 2 2 0 0 0 -1 0 0 0 0 1 1 1 1 0 0 1 4 2 1 1 1 1 1 1 1 256 256 256 "
  },

  {"Conv2d_bp_input_binary_get_cin1_factor", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 1, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {128, 64, 16, 16}, {1024, 512, 2, 2}, {16, 64, 28, 28, 16}, {16, 1024, 28, 28}, {16, 512, 56, 56}, {16, 512, 56, 56},
    ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    {1, 1, 2, 2}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", "",
    true, true, 32, 22211223, "128 64 16 64 28 28 512 32 56 56 2 2 1 32 1 0 0 0 0 2 2 0 0 0 0 1 1 0 0 16 1 2 1 1 1 32 2 28 8 1 16 4 4 8 2 143360 65536 57344 "
  },
  {"Conv2d_bp_input_binary_get_cin1_factor_NHWC", R"({"_pattern": "Conv2d_backprop_input", "tiling_type": "binary", "block_dim": {"CORE_NUM": 32}, "aub_num": 1, "cub_num": 2, "ub_size": 262000, "binary_mode": 2})",
    {4}, {128, 64, 16, 16}, {2, 2, 512, 1024}, {16, 64, 28, 28, 16}, {16, 28, 28, 1024}, {16, 512, 56, 56}, {16, 56, 56, 512},
    ge::FORMAT_ND, ge::FORMAT_HWCN, ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
    {1, 2, 2, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NHWC", "",
    true, true, 32, 22211223, "128 64 16 64 28 28 512 32 56 56 2 2 1 32 1 0 0 0 0 2 2 0 0 0 0 1 1 0 0 16 1 2 1 1 1 32 2 28 8 1 16 4 4 8 2 143360 65536 57344 "
  },
};

INSTANTIATE_TEST_CASE_P(Dx, Conv2DBackpropInputTilingRunTime2, testing::ValuesIn(general_cases_params));
