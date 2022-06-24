/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include <benchmark/benchmark.h>
#include "kernel_run_context_facker.h"
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#define private public
#include "op_log.h"
#include "op_tiling/op_tiling_util.h"
#include "runtime/trans_data.h"

using namespace std;
using namespace ge;

namespace optiling {
struct Runtime2TestParam {
  std::vector<std::string> attrs;
  std::vector<bool> input_const;
  std::vector<uint32_t> irnum;
};

static const std::map<std::string, ge::AnyValue::ValueType> kAttrTypesMap = {
    {"VT_INT", ge::AnyValue::ValueType::VT_INT},
    {"VT_BOOL", ge::AnyValue::ValueType::VT_BOOL},
    {"VT_FLOAT", ge::AnyValue::ValueType::VT_FLOAT},
    {"VT_STRING", ge::AnyValue::ValueType::VT_STRING},
    {"VT_LIST_INT", ge::AnyValue::ValueType::VT_LIST_INT},
    {"VT_LIST_BOOL", ge::AnyValue::ValueType::VT_LIST_BOOL},
    {"VT_LIST_FLOAT", ge::AnyValue::ValueType::VT_LIST_FLOAT},
    {"VT_LIST_LIST_INT", ge::AnyValue::ValueType::VT_LIST_LIST_INT},
};

const std::map<std::string, ge::Format> string_to_format_map = {{"NCHW", FORMAT_NCHW},
                                                                {"NHWC", FORMAT_NHWC},
                                                                {"NCDHW", FORMAT_NCDHW},
                                                                {"HWCN", FORMAT_HWCN},
                                                                {"DHWCN", FORMAT_DHWCN},
                                                                {"NDHWC", FORMAT_NDHWC},
                                                                {"CHWN", FORMAT_CHWN},
                                                                {"ND", FORMAT_ND},
                                                                {"NC1HWC0", FORMAT_NC1HWC0},
                                                                {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
                                                                {"FRACTAL_Z", FORMAT_FRACTAL_Z},
                                                                {"FRACTAL_ZN", FORMAT_FRACTAL_Z},
                                                                {"FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
                                                                {"NDC1HWC0", FORMAT_NDC1HWC0}};

static ge::Format StringToFormat(std::string format_string) {
  auto find_it = string_to_format_map.find(format_string);
  if (find_it != string_to_format_map.end()) {
    return find_it->second;
  }
  return FORMAT_ND;
}

static DataType StringToDtype(std::string dtype_string) {
  auto find_it = optiling::STR_TO_DATATYPE.find(dtype_string);
  if (find_it != optiling::STR_TO_DATATYPE.end()) {
    return find_it->second;
  }
  return ge::DT_FLOAT16;
}

static void add_input_desc_by_idx(Operator& op, int64_t idx, std::vector<int64_t> input_shape,
                                  std::vector<int64_t> input_ori_shape, std::string data_dtype, std::string src_format,
                                  std::string src_ori_format) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  op_info->MutableInputDesc(idx)->SetShape(GeShape(input_shape));
  op_info->MutableInputDesc(idx)->SetOriginShape(GeShape(input_ori_shape));
  op_info->MutableInputDesc(idx)->SetFormat(StringToFormat(src_format));
  op_info->MutableInputDesc(idx)->SetOriginFormat(StringToFormat(src_ori_format));
  op_info->MutableInputDesc(idx)->SetDataType(StringToDtype(data_dtype));
}

static void add_output_desc_by_idx(Operator& op, int64_t idx, std::vector<int64_t> input_shape,
                                   std::vector<int64_t> input_ori_shape, std::string data_dtype, std::string src_format,
                                   std::string src_ori_format) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  op_info->MutableOutputDesc(idx)->SetShape(GeShape(input_shape));
  op_info->MutableOutputDesc(idx)->SetOriginShape(GeShape(input_ori_shape));
  op_info->MutableOutputDesc(idx)->SetFormat(StringToFormat(src_format));
  op_info->MutableOutputDesc(idx)->SetOriginFormat(StringToFormat(src_ori_format));
  op_info->MutableOutputDesc(idx)->SetDataType(StringToDtype(data_dtype));
}

void TilingParseBenchmark(const std::string optype, std::string json_str, void *compile_info, benchmark::State& state) {
  auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling_parse;
  char *js_buf = new char[1 + json_str.length()];
  strcpy(js_buf, json_str.c_str());
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({js_buf})
                    .Outputs({compile_info})
                    .Build();
  for (auto _ : state) {
    tiling_prepare_func(holder.GetContext<gert::KernelContext>());
  }
  delete[] js_buf;
}

void TilingParse(const std::string optype, std::string json_str, void *compile_info) {
  auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling_parse;
  char *js_buf = new char[1 + json_str.length()];
  strcpy(js_buf, json_str.c_str());
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({js_buf})
                    .Outputs({compile_info})
                    .Build();
  tiling_prepare_func(holder.GetContext<gert::KernelContext>());
  delete[] js_buf;
}

static uint8_t* GetConstTensor(ge::Operator& op, const size_t index) {
  ge::Tensor const_tensor;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_name = op_desc->GetInputNameByIndex(index);
  if (op.GetInputConstData(input_name.c_str(), const_tensor) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  auto size = const_tensor.GetSize();
  auto data = const_tensor.GetData();
  ge::DataType const_dtype = op_desc->MutableInputDesc(index)->GetDataType();
  uint8_t* input_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder);
  std::memcpy(input_tensor + 1, data, size);
  int64_t value_size =
      (const_dtype == ge::DT_INT64 || const_dtype == ge::DT_UINT64) ? size / sizeof(int64_t) : size / sizeof(int32_t);
  gert::Tensor tensor({{value_size}, {value_size}},        // shape
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                      gert::kFollowing,                    // placement
                      const_dtype,                         // dt
                      nullptr);
  std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
  return input_tensor_holder;
}

void TilingBenchmark(ge::Operator& op, const Runtime2TestParam& param, void *info_base,
                     size_t tiling_len, benchmark::State& state) {
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);
  std::string optype = op.GetOpType();
  size_t input_size = op.GetInputsSize();
  std::vector<std::string> attrs = param.attrs;
  std::vector<bool> input_const = param.input_const;
  std::vector<uint32_t> irnum = param.irnum;
  if (irnum.size() > 0) {
    if (input_const.size() == 0) input_const.assign(irnum.size(), false);
  } else if (input_const.size() > 0) {
    if (irnum.size() == 0) irnum.assign(input_const.size(), 1);
  } else {
    input_const.assign(input_size, false);
    irnum.assign(input_size, 1);
  }
  size_t output_size = op.GetOutputsSize();

  // tiling data
  std::unique_ptr<uint8_t[]> tiling_data = gert::TilingData::CreateCap(tiling_len);
  auto faker = gert::TilingContextFaker()
                    .NodeIoNum(input_size, output_size)
                    .TilingData(tiling_data.get())
                    .CompileInfo(info_base)
                    .IrInstanceNum(irnum);

  vector<uint8_t*> const_tensors;
  std::vector<gert::StorageShape> input_shapes(input_size);
  std::vector<void *> input_shapes_ref(input_size);
  if (input_size > 0) {
    auto operator_info = OpDescUtils::GetOpDescFromOperator(op);
    if (operator_info == nullptr) return;
    size_t count = 0;
    for (size_t i = 0; i < input_const.size(); ++i) {
      if (input_const[i]) {
        uint8_t* input_tensor_holder = GetConstTensor(op, i);
        if (input_tensor_holder == nullptr) return;
        input_shapes_ref[count] = input_tensor_holder;
        const_tensors.push_back(input_tensor_holder);
        count++;
      } else for (uint32_t idx = 0; idx < irnum[i]; idx++) {
        auto input_desc = operator_info->MutableInputDesc(i + idx);
        if (input_desc == nullptr) continue;
        ge::Format input_format = input_desc->GetFormat();
        ge::Format origin_format = input_desc->GetOriginFormat();
        ge::DataType dtype = input_desc->GetDataType();
        faker = faker.NodeInputTd(count, dtype, origin_format, input_format);
        for (int64_t dim : input_desc->GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc->MutableShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }
        input_shapes_ref[count] = &input_shapes[count];
        count++;
      }
    }
    faker = faker.InputShapes(input_shapes_ref);
  }

  std::vector<gert::StorageShape> output_shapes(output_size);
  std::vector<void *> output_shapes_ref(output_size);
  if (output_size > 0) {
    for (size_t i = 0; i < output_size; ++i) {
      auto output_desc = operator_info->MutableOutputDesc(i);
      if (output_desc == nullptr) return;
      // get and check input format and shape
      ge::Format input_format = output_desc->GetFormat();
      ge::Format origin_format = output_desc->GetOriginFormat();
      ge::DataType dtype = output_desc->GetDataType();
      faker = faker.NodeOutputTd(i, dtype, origin_format, input_format);
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {
        output_shapes[i].MutableOriginShape().AppendDim(dim);
      }
      for (int64_t dim : output_desc->MutableShape().GetDims()) {
        output_shapes[i].MutableStorageShape().AppendDim(dim);
      }
      output_shapes_ref[i] = &output_shapes[i];
    }
    faker = faker.OutputShapes(output_shapes_ref);
  }

  auto op_attrs_map = op.GetAllAttrNamesAndTypes();
  std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value;
  std::pair<std::string, ge::AnyValue> p;
  if (attrs.size() > 0) {
    for (auto item : attrs) {
      p.first = item;
      auto attr_it = op_attrs_map.find(item);
      if (attr_it != op_attrs_map.end()) {
        auto type_it = kAttrTypesMap.find(attr_it->second);
        if (type_it != kAttrTypesMap.end()) {
          switch (type_it->second) {
            case ge::AnyValue::ValueType::VT_BOOL: {
              bool value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<bool>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_INT: {
              int64_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<int64_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_FLOAT: {
              float32_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<float32_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_STRING: {
              std::string value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<std::string>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_INT: {
              std::vector<int64_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<std::vector<int64_t>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_BOOL: {
              std::vector<bool> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<std::vector<bool>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_LIST_INT: {
              std::vector<std::vector<int64_t>> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return;
              p.second = ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(value);
            }
            break;
          }
        }
      }
      keys_to_value.push_back(p);
    }
    faker = faker.NodeAttrs(keys_to_value);
  }

  auto holder = faker.Build();

  if (gert::OpImplRegistry::GetInstance().GetOpImpl(optype) == nullptr) return;
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling;
  if (tiling_func == nullptr) return;
  gert::TilingContext *context = holder.GetContext<gert::TilingContext>();
  if (context == nullptr) return;
  for (auto _ : state) {
    tiling_func(context);
  }
}

static void tiling_parse_benchmark(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape,
                     std::string data_dtype, std::string src_format, std::string dst_format, std::string compile_json,
                     const int32_t &tiling_len, benchmark::State& state) {
  auto test_op = op::TransData("TransData");
  add_input_desc_by_idx(test_op, 0, input_shape, input_shape, data_dtype, src_format, src_format);
  add_output_desc_by_idx(test_op, 0, output_shape, output_shape, data_dtype, dst_format, dst_format);
  optiling::TransDataCompileInfo compile_info;
  TilingParseBenchmark("TransData", compile_json, &compile_info, state);
}

static void tiling_benchmark(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape,
                     std::string data_dtype, std::string src_format, std::string dst_format, std::string compile_json,
                     const int32_t &tiling_len, benchmark::State& state) {
  auto test_op = op::TransData("TransData");
  add_input_desc_by_idx(test_op, 0, input_shape, input_shape, data_dtype, src_format, src_format);
  add_output_desc_by_idx(test_op, 0, output_shape, output_shape, data_dtype, dst_format, dst_format);
  optiling::TransDataCompileInfo compile_info;
  TilingParse("TransData", compile_json, &compile_info);
  Runtime2TestParam param;
  TilingBenchmark(test_op, param, &compile_info, tiling_len, state);
}

static void TransDataTilingBenchmarkTiling1(benchmark::State& state) {
  std::vector<int64_t> input_shape = {1, 16, 7, 7};
  std::vector<int64_t> output_shape = {1, 1, 7, 7, 16};
  std::string dtype = "float16";
  std::string src_format = "NCHW";
  std::string dst_format = "NC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  int32_t tiling_len = sizeof(optiling::TransDataNtc100Param);
  tiling_parse_benchmark(input_shape, output_shape, dtype, src_format, dst_format, compile_info, tiling_len, state);
}
BENCHMARK(TransDataTilingBenchmarkTiling1);

static void TransDataTilingBenchmarkTiling2(benchmark::State& state) {
  std::vector<int64_t> input_shape = {1, 16, 7, 7};
  std::vector<int64_t> output_shape = {1, 1, 7, 7, 16};
  std::string dtype = "float16";
  std::string src_format = "NCHW";
  std::string dst_format = "NC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  int32_t tiling_len = sizeof(optiling::TransDataNtc100Param);
  tiling_benchmark(input_shape, output_shape, dtype, src_format, dst_format, compile_info, tiling_len, state);
}
BENCHMARK(TransDataTilingBenchmarkTiling2);

static void TransDataTilingBenchmarkTiling3(benchmark::State& state) {
  std::vector<int64_t> input_shape = {1, 1, 9, 16, 16};
  std::vector<int64_t> output_shape = {1, 16, 144};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_NZ";
  std::string dst_format = "ND";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_NZ\", \"dstFormat\": \"ND\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  int32_t tiling_len = sizeof(optiling::TransDataTc201Param);
  tiling_benchmark(input_shape, output_shape, dtype, src_format, dst_format, compile_info, tiling_len, state);
}
BENCHMARK(TransDataTilingBenchmarkTiling3);
}  // namespace optiling
