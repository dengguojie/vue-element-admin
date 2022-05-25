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
#include "trans_data.h"

using namespace gert;

namespace optiling {
namespace transdata {
int64_t GetC0SizeWithType(ge::DataType& dtype) {
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    return C0_32;
  }
  return C0_16;
}

int64_t GetShapeSize(const gert::Shape& in_shape, int32_t pos) {
  int32_t n = in_shape.GetDimNum();
  int64_t shape_size = 1;
  if (pos < 0) {
    pos = n + pos;
  }
  for (int32_t i = pos; i < n; i++) {
    shape_size *= in_shape[i];
  }
  return shape_size;
}

bool ConvertShapeHNC2HCNT(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                          gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  if (in_shape.GetDimNum() == 1) {
    axis_h = 1;
    axis_n = 1;
    axis_c = in_shape[0];
  } else if (in_shape.GetDimNum() == DIM_NUM_2) {
    axis_h = 1;
    axis_n = in_shape[0];
    axis_c = in_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < in_shape.GetDimNum() - DIM_NUM_2; i++) {
      shape_size *= in_shape[i];
    }
    axis_h = shape_size;
    axis_n = in_shape[in_shape.GetDimNum() + DIM_IDX_NEG_TWO];
    axis_c = in_shape[in_shape.GetDimNum() + DIM_IDX_NEG_ONE];
  }
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_n);
  in_shape_new.AppendDim(axis_c);
  int64_t axis_c0 = c0_size;
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(axis_no * axis_ni);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

bool Convert5HDToNCHW(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() < SHAPE_LEN_5D) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Convert5HDToNCHW failed, in_shape size < 5");
      return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_4]);

  int64_t axis_c = out_shape[1];
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(axis_c);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  return true;
}

bool Convert5HDToNHWC(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != SHAPE_LEN_5D || out_shape.GetDimNum() != SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING(
        "TransData", "Convert5HDToNHWC failed, The input shape dimension size should be 5 and output's should be 4!");
    return false;
  }
  int64_t axis_n = in_shape[0];
  int64_t axis_c1 = in_shape[1];
  int64_t axis_h = in_shape[DIM_IDX_2];
  int64_t axis_w = in_shape[DIM_IDX_3];
  int64_t axis_c0 = in_shape[DIM_IDX_4];
  int64_t axis_c = out_shape[out_shape.GetDimNum() - 1];
  int64_t axis_hw = axis_h * axis_w;
  in_shape_new = {axis_n, axis_c1, axis_hw, axis_c0};
  out_shape_new = {axis_n, axis_hw, axis_c};
  return true;
}

bool ConvertShapeHCNT2HNC(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                          gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  int64_t axis_n;
  int64_t axis_h;
  int64_t axis_c;
  size_t out_shape_size = out_shape.GetDimNum();
  if (out_shape_size == 1) {
    axis_h = 1;
    axis_n = 1;
    axis_c = out_shape[0];
  } else if (out_shape_size == DIM_NUM_2) {
    axis_h = 1;
    axis_n = out_shape[0];
    axis_c = out_shape[1];
  } else {
    int64_t shape_size = 1;
    for (size_t i = 0; i < out_shape_size - DIM_NUM_2; i++) {
      shape_size *= out_shape[i];
    }
    axis_h = shape_size;
    axis_n = out_shape[out_shape_size + DIM_IDX_NEG_TWO];
    axis_c = out_shape[out_shape_size + DIM_IDX_NEG_ONE];
  }
  int64_t axis_c0 = in_shape[in_shape.GetDimNum() - 1];
  int64_t axis_c1 = ge::CeilDiv(axis_c, axis_c0);
  int64_t axis_ni = NI_16;
  int64_t axis_no = ge::CeilDiv(axis_n, axis_ni);
  in_shape_new.AppendDim(axis_h);
  in_shape_new.AppendDim(axis_c1);
  in_shape_new.AppendDim(axis_no * axis_ni);
  in_shape_new.AppendDim(axis_c0);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c);
  return true;
}

bool ConvertNCHWTo5HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  if (in_shape.GetDimNum() != transdata::SHAPE_LEN_4D) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData",
                                    "ConvertNCHWTo5HD failed, The input shape dimension size is not correct!");
    return false;
  }
  in_shape_new.AppendDim(in_shape[0]);
  in_shape_new.AppendDim(in_shape[1]);
  in_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  int64_t c_idx = 1;
  int64_t axis_c1 = ge::CeilDiv(in_shape[c_idx], c0_size);
  out_shape_new.AppendDim(in_shape[0]);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(in_shape[DIM_IDX_2] * in_shape[DIM_IDX_3]);
  out_shape_new.AppendDim(c0_size);
  return true;
}

bool ConvertNHWCTo5HD(const gert::Shape& in_shape, const gert::Shape& out_shape, int64_t c0_size,
                      gert::Shape& in_shape_new, gert::Shape& out_shape_new) {
  static const int64_t HW_IDX = 1;
  static const int64_t C_IDX = 3;
  if (in_shape.GetDimNum() < 1) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "ConvertNHWCTo5HD failed, in_shape size < 1");
    return false;
  }
  for (int64_t i = 0; i < HW_IDX; i++) {
    in_shape_new.AppendDim(in_shape[i]);
  }
  int64_t n = in_shape.GetDimNum() - 1;
  int64_t shape_size = 1;
  for (int64_t i = HW_IDX; i < n; i++) {
    shape_size *= in_shape[i];
  }
  in_shape_new.AppendDim(shape_size);
  in_shape_new.AppendDim(in_shape[in_shape.GetDimNum() - 1]);
  int64_t axis_c1 = ge::CeilDiv(in_shape[C_IDX], c0_size);
  int64_t axis_n = in_shape[0];
  int64_t axis_h = in_shape_new[HW_IDX];
  int64_t axis_c0 = c0_size;
  out_shape_new.AppendDim(axis_n);
  out_shape_new.AppendDim(axis_c1);
  out_shape_new.AppendDim(axis_h);
  out_shape_new.AppendDim(axis_c0);
  return true;
}

size_t format_nd_array[ARRAYNDLEN] = {ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC};
size_t format_4d_array[ARRAY4DLEN] = {ge::FORMAT_NCHW, ge::FORMAT_NHWC};
RealShapeConvertFunc GetRealShapeConvertFunc(ge::Format src_format, ge::Format dst_format) {
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, RealShapeConvertFunc>(nullptr)
                          .Add(format_nd_array, ARRAYNDLEN, ge::FORMAT_FRACTAL_NZ, ConvertShapeHNC2HCNT)
                          .Add(ge::FORMAT_FRACTAL_NZ, format_nd_array, ARRAYNDLEN, ConvertShapeHCNT2HNC)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ConvertNCHWTo5HD)
                          .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, ConvertNHWCTo5HD)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, Convert5HDToNCHW)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, Convert5HDToNHWC);
  return table.Find(src_format, dst_format);
}

struct RealSrcDstFormat {
  RealSrcDstFormat() = default;
  RealSrcDstFormat(const RealSrcDstFormat& other) = default;
  RealSrcDstFormat& operator=(const RealSrcDstFormat& other) = default;
  RealSrcDstFormat(RealFormat src, RealFormat dst) : src(src), dst(dst) {
  }
  RealFormat src;
  RealFormat dst;
};

const RealSrcDstFormat* GetRealFormat(ge::Format src_format, ge::Format dst_format) {
  static RealSrcDstFormat default_value = {RF_END, RF_END};
  static auto table = TableDriven2<ge::FORMAT_END, ge::FORMAT_END, RealSrcDstFormat>(default_value)
                          .Add(format_nd_array, ARRAYNDLEN, ge::FORMAT_FRACTAL_NZ, RF_HNC, RF_HCNT)
                          .Add(ge::FORMAT_FRACTAL_NZ, format_nd_array, ARRAYNDLEN, RF_HCNT, RF_HNC)
                          .Add(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, RF_NCH, RF_NCHT)
                          .Add(ge::FORMAT_NHWC, ge::FORMAT_NC1HWC0, RF_NHC, RF_NCHT)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, RF_NCHT, RF_NCH)
                          .Add(ge::FORMAT_NC1HWC0, ge::FORMAT_NHWC, RF_NCHT, RF_NHC);

  return table.FindPointer(src_format, dst_format);
}
}  // namespace transdata

bool CheckShape(const gert::Shape &out_shape) {
  int32_t out_dims = out_shape.GetDimNum();
  for (int32_t i = 0; i < out_dims; i++) {
    if (out_shape[i] <= 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("TransData","CheckTensorShape, out_shape[%d] must be > 0", i);
      return false;
    }
  }
  return true;
}

ge::graphStatus TilingForTransData(TilingContext* context) {
  auto in_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, in_shape)
  auto out_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape)
  auto compile_info = reinterpret_cast<const TransDataCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info)
  auto src_td = context->GetInputDesc(0);
  auto dst_td = context->GetOutputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td)
  OPS_CHECK_NULL_WITH_CONTEXT(context, dst_td)

  bool check_ret =  CheckShape(out_shape->GetStorageShape());
  OP_TILING_CHECK(!check_ret, VECTOR_INNER_ERR_REPORT_TILIING("TransData", "CheckShape failed"),
                  return ge::GRAPH_FAILED);

  auto src_format = src_td->GetStorageFormat();
  auto dst_format = dst_td->GetStorageFormat();

  auto real_formats = transdata::GetRealFormat(src_format, dst_format);
  auto real_shape_converter = transdata::GetRealShapeConvertFunc(src_format, dst_format);
  if (real_formats == nullptr || real_shape_converter == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "real_formats or real_shape_converter nullptr!");
    return ge::FAILED;
  }

  auto data_type = src_td->GetDataType();
  auto c0_size = transdata::GetC0SizeWithType(data_type);
  gert::Shape real_in_shape{}, real_out_shape{};

  auto ret = real_shape_converter(in_shape->GetStorageShape(), out_shape->GetStorageShape(), c0_size, real_in_shape,
                                  real_out_shape);
  if (!ret) {
    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "real_shape_converter failed!");
    return ret;
  }
  int64_t block_elem_cnt = transdata::BLOCK_BYTE_SIZE / GetSizeByDataType(data_type);

  context->SetBlockDim(compile_info->block_dim);

  if ((real_formats->src == transdata::RF_HNC && real_formats->dst == transdata::RF_HCNT) ||
      (real_formats->src == transdata::RF_NHC && real_formats->dst == transdata::RF_NCHT)) {
    return transdata::TillingPositiveMode1010(context, real_in_shape, real_out_shape, real_formats->src,
                                              real_formats->dst, compile_info->block_dim, block_elem_cnt,
                                              compile_info->ub_size);
  }
  if ((real_formats->src == transdata::RF_HCNT && real_formats->dst == transdata::RF_HNC) ||
      (real_formats->src == transdata::RF_NCHT && real_formats->dst == transdata::RF_NHC)) {
    return transdata::TilingNegativeTc201(context, real_in_shape, real_out_shape, real_formats->src, real_formats->dst,
                                          compile_info->block_dim, block_elem_cnt, compile_info->ub_size, data_type);
  }
  if (real_formats->src == transdata::RF_NCHT && real_formats->dst == transdata::RF_NCH) {
    return transdata::TilingNegativeNtc200(context, real_in_shape, real_out_shape, real_formats->src, real_formats->dst,
                                           compile_info->block_dim, block_elem_cnt, compile_info->ub_size, data_type,
                                           compile_info->vnc_fp32_flag);
  }
  if (real_formats->src == transdata::RF_NCH && real_formats->dst == transdata::RF_NCHT) {
    return transdata::TilingPositiveSourceNtc100(context, real_in_shape, real_out_shape, real_formats->src,
                                                 real_formats->dst, compile_info->block_dim, block_elem_cnt,
                                                 compile_info->ub_size, data_type, c0_size);
  }
  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Unsupported src and dst format %d -> %d", real_formats->src,
                                  real_formats->dst);
  return ge::GRAPH_FAILED;
}

bool CheckCompileInfo(TransDataCompileInfo* compile_info) {
  OP_LOGD("TransData", "Parsed TransData compile info(ub_size, block_dim, group, vnc_fp32_flag): %ld, %ld, %ld, %ld",
          compile_info->ub_size, compile_info->block_dim, compile_info->group, compile_info->vnc_fp32_flag);
  if(compile_info->ub_size <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "invalid value ub_size <= 0");
    return false;
  }
  if(compile_info->block_dim <= 0) {
    VECTOR_INNER_ERR_REPORT_TILIING("TransData", "invalid value block_dim <= 0");
    return false;
  }

  return true;
}

ge::graphStatus TilingPrepareForTransdata(TilingParseContext* context) {
  auto compile_info = MutableCompileInfo<TransDataCompileInfo>(context);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OP_TILING_CHECK(compile_info == nullptr || parsed_object_cinfo == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING("TransData", "compile_info or json_str nullptr!"),
                  return ge::GRAPH_FAILED);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING("TransData", "get vars failed."),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!GetCompileValue(vars, "ub_size", compile_info->ub_size),
                  VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Failed to get ub_size in compile_info"),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!GetCompileValue(vars, "block_dim", compile_info->block_dim),
                  VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Failed to get block_dim in compile_info"),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!GetCompileValue(vars, "group", compile_info->group),
                  VECTOR_INNER_ERR_REPORT_TILIING("TransData", "Failed to get group in compile_info"),
                  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!GetCompileValue(vars, "vnc_fp32_flag", compile_info->vnc_fp32_flag), compile_info->vnc_fp32_flag = 0,
                  OP_LOGI("TransData", "Can not find vnc_fp32_flag in json, use default value 0"));

  bool check_ret = CheckCompileInfo(compile_info);
  OP_TILING_CHECK(!check_ret, VECTOR_INNER_ERR_REPORT_TILIING("TransData", "CheckCompileInfo failed"),
                  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(TransData).Tiling(TilingForTransData).TilingParse<TransDataCompileInfo>(TilingPrepareForTransdata);
}  // namespace optiling
