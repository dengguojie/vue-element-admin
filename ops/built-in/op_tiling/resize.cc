/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

/*!
 * \file resize.cc
 * \brief
 */
#include "resize_common.h"


namespace optiling {
struct ResizeNearest3DTilingParams
{
    int64_t tiling_mode;
    int64_t batch_n;
    int64_t batch_c1;
    int64_t input_d;
    int64_t input_h;
    int64_t input_w;
    int64_t output_d;
    int64_t output_h;
    int64_t output_w;
    int64_t avg_input;
    int64_t loop_input;
    int64_t tail_input;
    int64_t nd;
    int64_t avg_nd;
    int64_t last_nd;
    int64_t block_num;
    int64_t loop_h;
    int64_t tail_h;
    int64_t loop_w;
    int64_t tail_w;
    int64_t move_c1;
    int64_t loop_c1;
    int64_t tail_c1;
};

void InitTilingParamsNearest(ResizeNearest3DTilingParams& params) {
    params.tiling_mode = 0;
    params.batch_n = 0;
    params.batch_c1 = 0;
    params.input_d = 0;
    params.input_h = 0;
    params.input_w = 0;
    params.output_d = 0;
    params.output_h = 0;
    params.output_w = 0;
    params.avg_input = 0;
    params.loop_input = 0;
    params.tail_input = 0;
    params.nd = 0;
    params.avg_nd = 0;
    params.last_nd = 0;
    params.block_num = 0;
    params.loop_h = 0;
    params.tail_h = 0;
    params.loop_w = 0;
    params.tail_w = 0;
    params.move_c1 = 0;
    params.loop_c1 = 0;
    params.tail_c1 = 0;
}

bool ResizeNearest3DParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                              ResizeCommonInputCompile& compile_value) {
  if (compile_info.count("vars") == 0) {
    return false;
  }
  const nlohmann::json& all_vars = compile_info["vars"];

  OP_TILING_CHECK(!GetCompileValue(all_vars, "core_num", compile_value.core_num2),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeNearest3DParseFunc, get core_num error"),
                  return false);
  OP_TILING_CHECK(!GetCompileValue(all_vars, "left_w", compile_value.left_w),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeNearest3DParseFunc, get left_w error"),
                  return false);

  return true;
}

int64_t CeilDivNearest(int64_t num0, int64_t num1) {
    int64_t res = 0;
    if (num1 == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("ResizeNearest3DTiling", "CeilDivNearest, num1 cannot be zero");
      return 0;
    }
    res = (num0 + num1 - 1) / num1;
    return res;
}

int64_t CalTailNearest(int64_t num0, int64_t num1) {
    int64_t res = 0;
    if (num1 == 0) {
      VECTOR_INNER_ERR_REPORT_TILIING("ResizeNearest3DTiling", "CalTailNearest, num1 cannot be zero");
      return 0;
    }
    res = num0 % num1;
    if (res == 0) {
        res = num1;
    }
    return res;
}

int64_t NearestTilingMode(ResizeNearest3DTilingParams& params, int64_t left_w) {
    int64_t input_d = params.input_d;
    int64_t input_h = params.input_h;
    int64_t input_w = params.input_w;
    int64_t output_h = params.output_h;
    int64_t output_w = params.output_w;
    int64_t output_d = params.output_d;
    int64_t c1 = params.batch_c1;
    int64_t out_wc1 = output_w * c1;
    int64_t in_wc1 = input_w * c1;
    int64_t move_c1 = 0;
    int64_t loop_c1 = 0;
    int64_t tail_c1 = 0;
	
	const int64_t max_out = 1024;
	const int64_t tiling_0 = 0;
	const int64_t tiling_1 = 1;
	const int64_t tiling_2 = 2;
	const int64_t tiling_3 = 3;
	const int64_t tiling_4 = 4;
	const int64_t tiling_5 = 5;
	const int64_t tiling_6 = 6;
	const int64_t tiling_7 = 7;
	const int64_t tiling_8 = 8;

    if (input_h == output_h && input_w == output_w && input_d == output_d) {
        return tiling_8;
    }
    if (out_wc1 <= max_out && in_wc1 <= max_out) {
       return tiling_0;
    }
    if (out_wc1 <= max_out && in_wc1 > max_out) {
       if (input_w <= max_out) {
          move_c1 = max_out / input_w;
          loop_c1 = CeilDivNearest(c1, move_c1);
          tail_c1 = CalTailNearest(c1, move_c1);
          params.move_c1 = move_c1;
          params.loop_c1 = loop_c1;
          params.tail_c1 = tail_c1;
          return tiling_1;
       } else {
         int64_t avg_input = max_out;
         int64_t loop_input = CeilDivNearest(input_w, avg_input);
         int64_t tail_input = CalTailNearest(input_w, avg_input);
         params.avg_input = avg_input;
         params.loop_input = loop_input;
         params.tail_input = tail_input;
         return tiling_2;
       }
    }
    if (out_wc1 > max_out && in_wc1 <= max_out) {
       if (output_w < max_out) {
          move_c1 = max_out / output_w;
          loop_c1 = CeilDivNearest(c1, move_c1);
          tail_c1 = CalTailNearest(c1, move_c1);
          params.move_c1 = move_c1;
          params.loop_c1 = loop_c1;
          params.tail_c1 = tail_c1;
          return tiling_3;
       }
    }
    if (output_w < max_out && input_w < left_w - output_w) {
        return tiling_4;
    }
    if (output_w < max_out && input_w >= left_w - output_w) {
        int64_t avg_input = left_w - output_w;
        int64_t loop_input = CeilDivNearest(input_w, avg_input);
        int64_t tail_input = CalTailNearest(input_w, avg_input);
        params.avg_input = avg_input;
        params.loop_input = loop_input;
        params.tail_input = tail_input;
        return tiling_5;
    }
    if (output_w >= max_out && input_w < left_w - max_out) {
        return tiling_6;
    } else {
        int64_t avg_input = left_w - max_out;
        int64_t loop_input = CeilDivNearest(input_w, avg_input);
        int64_t tail_input = CalTailNearest(input_w, avg_input);
        params.avg_input = avg_input;
        params.loop_input = loop_input;
        params.tail_input = tail_input;
        return tiling_7;
    }
}

void NearestRunningInfo(ResizeNearest3DTilingParams& tiling_params, std::vector<int64_t> images_shape,
                        int64_t core_num, int64_t left_w, int64_t output_d, int64_t output_h, int64_t output_w) {
    int64_t batch_n = images_shape[0];
    int64_t batch_c1 = images_shape[2];
    int64_t input_d = images_shape[1];
    int64_t input_h = images_shape[3];
    int64_t input_w = images_shape[4];
    int64_t nd = batch_n * output_d;
    int64_t avg_nd = CeilDivNearest(nd, core_num);
    int64_t block_num = CeilDivNearest(nd, avg_nd);
    int64_t last_nd = nd - avg_nd * (block_num - 1);
    int64_t loop_h = CeilDivNearest(output_h, 1024);
    int64_t tail_h = CalTailNearest(output_h, 1024);
    int64_t loop_w = CeilDivNearest(output_w, 1024);
    int64_t tail_w = CalTailNearest(output_w, 1024);

    tiling_params.batch_n = batch_n;
    tiling_params.batch_c1 = batch_c1;
    tiling_params.input_d = input_d;
    tiling_params.input_h = input_h;
    tiling_params.input_w = input_w;
    tiling_params.output_d = output_d;
    tiling_params.output_h = output_h;
    tiling_params.output_w = output_w;
    tiling_params.nd = nd;
    tiling_params.avg_nd = avg_nd;
    tiling_params.last_nd = last_nd;
    tiling_params.block_num = block_num;
    tiling_params.loop_h = loop_h;
    tiling_params.tail_h = tail_h;
    tiling_params.loop_w = loop_w;
    tiling_params.tail_w = tail_w;

    int64_t tiling_mode = NearestTilingMode(tiling_params, left_w);
    tiling_params.tiling_mode = tiling_mode;
}

void SetNearestInfo(const ResizeNearest3DTilingParams& tiling_params, utils::OpRunInfo& run_info) {
    run_info.AddTilingData(tiling_params.tiling_mode);
    run_info.AddTilingData(tiling_params.batch_n);
    run_info.AddTilingData(tiling_params.batch_c1);
    run_info.AddTilingData(tiling_params.input_d);
    run_info.AddTilingData(tiling_params.input_h);
    run_info.AddTilingData(tiling_params.input_w);
    run_info.AddTilingData(tiling_params.output_d);
    run_info.AddTilingData(tiling_params.output_h);
    run_info.AddTilingData(tiling_params.output_w);
    run_info.AddTilingData(tiling_params.avg_input);
    run_info.AddTilingData(tiling_params.loop_input);
    run_info.AddTilingData(tiling_params.tail_input);
    run_info.AddTilingData(tiling_params.nd);
    run_info.AddTilingData(tiling_params.avg_nd);
    run_info.AddTilingData(tiling_params.last_nd);
    run_info.AddTilingData(tiling_params.block_num);
    run_info.AddTilingData(tiling_params.loop_h);
    run_info.AddTilingData(tiling_params.tail_h);
    run_info.AddTilingData(tiling_params.loop_w);
    run_info.AddTilingData(tiling_params.tail_w);
    run_info.AddTilingData(tiling_params.move_c1);
    run_info.AddTilingData(tiling_params.loop_c1);
    run_info.AddTilingData(tiling_params.tail_c1);
}

void PrintNearestParams(const std::string& op_type, const ResizeNearest3DTilingParams& tiling_params) {
    OP_LOGD(op_type, "tiling_mode=%d.", tiling_params.tiling_mode);
    OP_LOGD(op_type, "batch_n=%d.", tiling_params.batch_n);
    OP_LOGD(op_type, "batch_c1=%d.", tiling_params.batch_c1);
    OP_LOGD(op_type, "input_d=%d.", tiling_params.input_d);
    OP_LOGD(op_type, "input_h=%d.", tiling_params.input_h);
    OP_LOGD(op_type, "input_w=%d.", tiling_params.input_w);
    OP_LOGD(op_type, "output_d=%d.", tiling_params.output_d);
    OP_LOGD(op_type, "output_h=%d.", tiling_params.output_h);
    OP_LOGD(op_type, "output_w=%d.", tiling_params.output_w);
    OP_LOGD(op_type, "avg_input=%d.", tiling_params.avg_input);
    OP_LOGD(op_type, "loop_input=%d.", tiling_params.loop_input);
    OP_LOGD(op_type, "tail_input=%d.", tiling_params.tail_input);
    OP_LOGD(op_type, "nd=%d.", tiling_params.nd);
    OP_LOGD(op_type, "avg_nd=%d.", tiling_params.avg_nd);
    OP_LOGD(op_type, "last_nd=%d.", tiling_params.last_nd);
    OP_LOGD(op_type, "block_num=%d.", tiling_params.block_num);
    OP_LOGD(op_type, "loop_h=%d.", tiling_params.loop_h);
    OP_LOGD(op_type, "tail_h=%d.", tiling_params.tail_h);
    OP_LOGD(op_type, "loop_w=%d.", tiling_params.loop_w);
    OP_LOGD(op_type, "tail_w=%d.", tiling_params.tail_w);
    OP_LOGD(op_type, "move_c1=%d.", tiling_params.move_c1);
    OP_LOGD(op_type, "loop_c1=%d.", tiling_params.loop_c1);
    OP_LOGD(op_type, "tail_c1=%d.", tiling_params.tail_c1);
}

bool ResizeNearest3DTiling(const std::string& op_type, const ge::Operator& op_paras,
                           const ResizeCommonInputCompile& op_info, utils::OpRunInfo& run_info) {
    int64_t core_num = op_info.core_num2;
    int64_t left_w = op_info.left_w;

    ResizeNearest3DTilingParams tiling_params;
    InitTilingParamsNearest(tiling_params);
    
    auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    auto images_desc = operator_info->MutableInputDesc(0);
    auto output_desc = operator_info->MutableOutputDesc(0);

    const std::vector<int64_t>& images_shape = images_desc->MutableShape().GetDims();
    const std::vector<int64_t>& output_shape = output_desc->MutableShape().GetDims();
    int64_t output_d = output_shape[1];
    int64_t output_h = output_shape[3];
    int64_t output_w = output_shape[4];

    NearestRunningInfo(tiling_params, images_shape, core_num, left_w, output_d, output_h, output_w);
    SetNearestInfo(tiling_params, run_info);
    PrintNearestParams(op_type, tiling_params);

    run_info.SetBlockDim(tiling_params.block_num);
    return true;
}

bool ResizeParseFunc(const std::string& op_type, const nlohmann::json& compile_info,
                     ResizeCommonInputCompile& compile_value) {
    const int64_t nearest_2d = 20;
    const int64_t linear_2d = 21;
    const int64_t nearest_3d = 22;
    if (compile_info.count("vars") == 0) {
        return false;
    }
    int64_t mode_name;
    const nlohmann::json& all_vars = compile_info["vars"];
    OP_TILING_CHECK(!GetCompileValue(all_vars, "mode_name", mode_name),
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type.c_str(), "ResizeParseFunc, get mode_name error"),
                    return false);
    std::string op_type_mode;
    if (mode_name == nearest_2d) {
        op_type_mode = "ResizeNearestNeighborV2";
        return ResizeCommonParseFunc(op_type_mode, compile_info, compile_value);
    } else if (mode_name == linear_2d) {
        op_type_mode = "ResizeBilinearV2";
        return ResizeCommonParseFunc(op_type_mode, compile_info, compile_value);
    } else if (mode_name == nearest_3d) {
        op_type_mode = "ResizeNearest3D";
        return ResizeNearest3DParseFunc(op_type_mode, compile_info, compile_value);
    } else {
        OP_LOGE(op_type.c_str(), "Mode only support nearest or linear.");
        return false;
    }
}
bool ResizeTiling(const std::string& op_type, const ge::Operator& op_paras,
                  const ResizeCommonInputCompile& op_info, utils::OpRunInfo& run_info) {
    using namespace ge;
    ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
    OP_TILING_CHECK(op_desc == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "The op_desc is nullptr."),
                    return false);
    std::string mode = "nearest";
    ge::AttrUtils::GetStr(op_desc, "mode", mode);

    ge::Format input_format = op_paras.GetInputDesc(0).GetFormat();
    if (input_format == FORMAT_NC1HWC0) {
      std::string op_type_2d;
      if (mode == "nearest") {
        op_type_2d = "ResizeNearestNeighborV2";
      } else if (mode == "linear") {
        op_type_2d = "ResizeBilinearV2";
      } else {
        OP_LOGE(op_type.c_str(), "Mode only support nearest or linear.");
        return false;
      }
      return ResizeCommonTiling(op_type_2d, op_paras, op_info, run_info);
    } else if (input_format == FORMAT_NDC1HWC0) {
      std::string op_type_3d;
      if (mode == "nearest") {
        op_type_3d = "ResizeNearest3D";
      } else {
        OP_LOGE(op_type.c_str(), "Mode only support nearest when 3d.");
        return false;
      }
      return ResizeNearest3DTiling(op_type_3d, op_paras, op_info, run_info);
    } else {
      OP_LOGE(op_type.c_str(), "Only support 2D and 3D resize.");
      return false;
    }
}
// register tiling interface of the Resize op.
REGISTER_OP_TILING_V3_CUSTOM(Resize, ResizeTiling, ResizeParseFunc, ResizeCommonInputCompile);
}  // namespace optiling.