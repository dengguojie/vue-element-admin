/*!
 * \file layer_norm.cpp
 * \brief dynamic shape tiling of layer_norm
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "../fusion_pass/common/fp16_t.hpp"
#include "../op_proto/util/error_util.h"
#include "error_log.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "op_tiling.h"
#include "vector_tiling_log.h"

namespace optiling {
const int32_t REDUCE_MEAN_COF_FP32 = 1;
const int32_t REDUCE_MEAN_COF_FP16 = 2;
const int32_t NUM_TW = 2;
const int32_t NUM_THR = 3;

struct TilingParams {
  /* data */
  int32_t block_dim;
  int32_t block_tiling_axis;
  int32_t block_tiling_axis_1 = 0;
  int32_t block_factor;
  int32_t block_factor_1 = 1;
  int32_t ub_tiling_axis;
  int32_t ub_factor;
  int32_t ub_tiling_axis_reduce;
  int32_t ub_fuse_factor;
};

struct CompileInfo {
  /* data */
  bool is_const = false;
  bool is_const_post = false;
  bool atomic = false;
  bool is_keep_dims = false;
  bool is_normal = true;
  bool is_fuse_axis = false;
  int64_t max_ub_count;
  int32_t core_num;
  int32_t min_block_size;
};

bool IsInVector(std::vector<int32_t> input, int32_t value) {
  for (uint32_t i = 0; i < input.size(); i++) {
    if (input[i] == value) {
      return true;
    }
  }
  return false;
}

int32_t
CalcPatternKey(std::vector<int64_t> input, std::vector<int32_t> reduce_axis,
               int32_t block_split_axis, int32_t ub_split_axis_index_reduce,
               int32_t ub_split_axis, bool is_normal, bool is_fuse_axis) {
  int32_t pattern = 0;

  for (size_t i = 0; i < input.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      pattern += NUM_THR * pow(NUM_TW, (input.size() - i - 1));
    } else {
      pattern += pow(NUM_TW, (input.size() - 1 - i));
    }
  }

  pattern +=
      block_split_axis * 100 + ub_split_axis * 10 + ub_split_axis_index_reduce;

  if (!is_normal) {
    pattern = pattern * NUM_TW;
  }
  if (!is_fuse_axis) {
    pattern = pattern * NUM_THR;
  }

  return pattern;
}

int32_t CalcTilingKey(CompileInfo &commoninfo, std::vector<int64_t> input_x,
                      TilingParams tilingparams,
                      std::vector<int32_t> reduce_axis) {
  int32_t key = 0;
  int32_t block_split_axis = tilingparams.block_tiling_axis;
  int32_t block_split_axis_1 = tilingparams.block_tiling_axis_1;
  int32_t ub_split_axis = tilingparams.ub_tiling_axis;
  int32_t ub_split_axis_index_reduce = tilingparams.ub_tiling_axis_reduce;
  bool is_normal = commoninfo.is_normal;
  bool is_fuse_axis = commoninfo.is_fuse_axis;
  int32_t pattern = CalcPatternKey(input_x, reduce_axis, block_split_axis,
                                   ub_split_axis_index_reduce, ub_split_axis,
                                   is_normal, is_fuse_axis);
  std::vector<int32_t> val = {1000000000, 10000000, 1000000,
                              100000,     10000,    1000};
  std::vector<int32_t> pos = {
      0, 0, block_split_axis, ub_split_axis, pattern, block_split_axis_1};
  for (size_t i = 0; i < pos.size(); i++) {
    key += pos[i] * val[i];
  }

  return key;
}

bool GetCompileInfo(const std::string &op_type, const nlohmann::json &op_info,
                    CompileInfo &compileinfo, std::vector<int32_t> reduce_axis,
                    std::vector<int64_t> input_shape, OpRunInfo &run_info) {
  std::vector<int32_t> common_info;
  std::vector<int32_t> pattern_info;
  std::vector<int32_t> ub_info;

  CHECK((op_info.find("common_info") != op_info.end()),
        "op [%s] : compile info not contain [common_info]", op_type.c_str());
  common_info = op_info.at("common_info").get<std::vector<int32_t>>();
  CHECK((op_info.find("pattern_info") != op_info.end()),
        "op [%s] : compile info not contain [pattern_info]", op_type.c_str());
  pattern_info = op_info.at("pattern_info").get<std::vector<int32_t>>();
  CHECK((op_info.find("ub_info") != op_info.end()),
        "op [%s] : compile info not contain [ub_info]", op_type.c_str());
  ub_info = op_info.at("ub_info").get<std::vector<int32_t>>();

  compileinfo.core_num = common_info[0];
  compileinfo.is_keep_dims = (bool)common_info[1];
  compileinfo.min_block_size = common_info[2];
  compileinfo.atomic = (bool)common_info[3];

  V_OP_TILING_CHECK(compileinfo.min_block_size > 0,
                    OP_LOGE(op_type.c_str(),
                            "min_block_size is %d that is illegal",
                            compileinfo.min_block_size),
                    return false);

  V_OP_TILING_CHECK(compileinfo.core_num > 0,
                    OP_LOGE(op_type.c_str(), "core_num is %d that is illegal",
                            compileinfo.core_num),
                    return false);

  float reduce_mean_cof = 1.0;

  if (op_info.count("reduce_mean_cof_dtype") > 0) {
    const std::string &reduce_mean_cof_dtype =
        op_info.at("reduce_mean_cof_dtype").get<std::string>();

    if (reduce_mean_cof_dtype == "float32") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
      ByteBufferPut(run_info.tiling_data, (float)reduce_mean_cof);
      OP_LOGD(op_type.c_str(), "fp32 reduce mean cof:%f", reduce_mean_cof);
    } else if (reduce_mean_cof_dtype == "float16") {
      for (uint32_t i = 0; i < input_shape.size(); i++) {
        if (IsInVector(reduce_axis, i)) {
          reduce_mean_cof = reduce_mean_cof / input_shape[i];
        }
      }
      fe::fp16_t reduce_mean_cof_fp16;
      reduce_mean_cof_fp16 = reduce_mean_cof;
      ByteBufferPut(run_info.tiling_data, (fe::fp16_t)reduce_mean_cof_fp16);
      OP_LOGD(op_type.c_str(), "fp16 reduce mean cof:%f", reduce_mean_cof);
    }
  }

  return true;
}

bool CheckWorkspaceCase(std::vector<int64_t> input_x,
                        std::vector<int32_t> reduce_axis, int32_t max_ub_size) {
  int32_t reduce_index = 0;
  int32_t reduce_shape_size = 1;
  for (size_t j = 0; j < reduce_axis.size(); j++) {
    reduce_index = reduce_axis[j];
    reduce_shape_size *= input_x[reduce_index];
  }
  if (reduce_shape_size > max_ub_size) {
    OP_LOGI("CheckWorkspaceCase: workspace case.");
    return true;
  }
  OP_LOGI("CheckWorkspaceCase: normal case.");
  return false;
}

bool CheckExceedUbSize(int32_t block_inner, std::vector<int64_t> input_x,
                       size_t i, int32_t max_ub_size) {
  // judge exceed ub_size  and do workspace

  int32_t shape_size = block_inner;
  for (size_t j = i + 1; j < input_x.size(); j++) {
    shape_size *= input_x[j];
  }
  if (shape_size > max_ub_size) {
    OP_LOGI("CheckExceedUbSize: true");
    return true;
  }
  OP_LOGI("CheckExceedUbSize: false");
  return false;
}

std::vector<int32_t> GetUbTilingData(int32_t block_inner, size_t i,
                                     std::vector<int64_t> input_x,
                                     int32_t max_ub_size,
                                     std::vector<int32_t> reduce_axis,
                                     int32_t block) {
  int32_t x_size = 1;
  int32_t ub_tiling_axis = i;
  int32_t ub_factor = 1;
  int32_t ub_outer = 1;
  int32_t ub_mul_num = 1;
  int32_t axis_num = 1;
  bool is_workspacecase = false;
  bool is_open_multi_core = false;

  for (size_t j = input_x.size() - 1; j > i; j--) {
    axis_num = 1;
    x_size *= input_x[j];
    if (!IsInVector(reduce_axis, j)) {
      ub_mul_num *= input_x[j];
      axis_num = input_x[j];
    }
    if (x_size > max_ub_size) {
      ub_tiling_axis = j;
      ub_mul_num = ub_mul_num / axis_num;

      ub_factor = max_ub_size / (x_size / input_x[j]);
      if ((ub_factor * ub_mul_num < block) && !IsInVector(reduce_axis, j)) {
        ub_tiling_axis = j + 1;
        ub_factor = input_x[ub_tiling_axis];
        if (!IsInVector(reduce_axis, ub_tiling_axis)) {
          ub_mul_num = ub_mul_num / ub_factor;
        }
      }
      if (IsInVector(reduce_axis, ub_tiling_axis)) {
        is_workspacecase = true;
      } else {
        // check input_x[j]%uf memery overflow
        int32_t min_uf = (block + ub_mul_num - 1) / ub_mul_num;
        for (int32_t uf = ub_factor; uf >= min_uf; uf--) {
          int32_t tail_region = input_x[ub_tiling_axis] % uf;
          if (!(input_x[ub_tiling_axis] % uf) ||
              (tail_region * ub_mul_num >= block)) {
            ub_factor = uf;
            is_open_multi_core = true;
            break;
          }
        }
      }
      ub_outer = input_x[ub_tiling_axis] % ub_factor == 0
                     ? input_x[ub_tiling_axis] / ub_factor
                     : (input_x[ub_tiling_axis] + ub_factor) / ub_factor;
      std::vector<int32_t> res = {ub_tiling_axis, ub_factor, is_workspacecase,
                                  is_open_multi_core, ub_outer};
      return res;
    }
  }
  x_size *= block_inner;
  if (x_size > max_ub_size) {
    int32_t split_num = block_inner;
    ub_factor = max_ub_size / (x_size / block_inner);
    if ((ub_factor * ub_mul_num < block) && !IsInVector(reduce_axis, i)) {
      ub_tiling_axis = i + 1;
      ub_factor = input_x[ub_tiling_axis];
      split_num = input_x[ub_tiling_axis];
      ub_mul_num = ub_mul_num / axis_num;
    }
    if (IsInVector(reduce_axis, ub_tiling_axis)) {
      is_workspacecase = true;
    } else {
      // check input_x[j]%uf memery overflow
      int32_t min_uf = (block + ub_mul_num - 1) / ub_mul_num;
      for (int32_t uf = ub_factor; uf >= min_uf; uf--) {
        int32_t tail_region = split_num % uf;
        if (!(split_num % uf) || (tail_region * ub_mul_num >= block)) {
          ub_factor = uf;
          is_open_multi_core = true;
          break;
        }
      }
    }
  }
  ub_outer = input_x[ub_tiling_axis] % ub_factor == 0
                 ? input_x[ub_tiling_axis] / ub_factor
                 : (input_x[ub_tiling_axis] + ub_factor) / ub_factor;
  std::vector<int32_t> res = {ub_tiling_axis, ub_factor, is_workspacecase,
                              is_open_multi_core, ub_outer};
  return res;
}

int32_t GetUbReduceAxis(std::vector<int64_t> input_x, const int32_t max_ub_size,
                        std::vector<int32_t> reduce_axis) {
  int32_t reduce_shape_size = 1;
  for (size_t i = reduce_axis.size() - 1; i >= 0; i--) {
    reduce_shape_size *= input_x[reduce_axis[i]];
    if (reduce_shape_size > max_ub_size) {
      OP_LOGI("In workspace case, ub axis must be reduce axis--> true");
      return i;
    }
  }
  OP_LOGI("In workspace case, ub axis must be reduce axis--> false");
  return 0;
}

int32_t GetUnblockAxisOutputMul(int32_t block_axis,
                                std::vector<int64_t> input_x,
                                std::vector<int32_t> reduce_axis) {
  int32_t mul_num = 1;
  for (int32_t i = 0; i < input_x.size(); i++) {
    if (!IsInVector(reduce_axis, i) && (block_axis != i)) {
      mul_num *= input_x[i];
    }
  }
  return mul_num;
}

std::vector<int32_t>
GetBlockTilingData(int32_t block_axis, int32_t block_factor,
                   int32_t ub_tiling_axis, int32_t ub_outer,
                   std::vector<int64_t> input_x,
                   std::vector<int32_t> reduce_axis, int32_t core_num) {
  int32_t block_axis_1 = block_axis + 1;
  block_factor = input_x[block_axis];
  int32_t block_factor_1 = 1;
  if (block_factor > core_num) {
    block_factor_1 = ub_outer;
  }
  int32_t block_axis_1_size = 1;
  int32_t fuse_axis = input_x[block_axis];
  std::vector<int32_t> cm_core_num_list;
  cm_core_num_list.push_back(core_num);
  for (int32_t i = core_num - 1; i > 1; i--) {
    if (!(core_num % i)) {
      cm_core_num_list.push_back(i);
    }
  }
  cm_core_num_list.push_back(1);
  for (int32_t idx = block_axis + 1; idx <= ub_tiling_axis; idx++) {
    if (IsInVector(reduce_axis, idx)) {
      continue;
    }
    if (idx == ub_tiling_axis) {
      fuse_axis *= ub_outer;
      block_axis_1_size = ub_outer;
    } else {
      fuse_axis *= input_x[idx];
      block_axis_1_size = input_x[idx];
    }

    if (!(fuse_axis % core_num)) {
      block_axis_1 = idx;
      for (size_t k = 0; k < cm_core_num_list.size(); k++) {
        if (!(input_x[block_axis] % cm_core_num_list[k])) {
          block_factor = cm_core_num_list[k];
          block_factor_1 = core_num / cm_core_num_list[k];
          break;
        } else if (!(block_axis_1_size % cm_core_num_list[k])) {
          block_factor_1 = cm_core_num_list[k];
          block_factor = core_num / cm_core_num_list[k];
          break;
        }
      }
      break;
    }
  }

  std::vector<int32_t> res = {block_axis, block_factor, block_axis_1,
                              block_factor_1};
  return res;
}

void GetTilingData(std::vector<int64_t> input_x, TilingParams &tilingparams,
                   std::vector<int32_t> reduce_axis, int32_t core_num,
                   int32_t max_ub_size, CompileInfo &compileinfo,
                   const std::string input_dtype) {
  // std::vector<int32_t> tiling_params;

  int32_t block = 1;
  int32_t fuse_axis_num = 1;
  if (input_dtype == "float32") {
    block = 8;
  } else {
    block = 16;
  }
  if (input_x.size() == reduce_axis.size()) {
    // all_reduce
    tilingparams.block_dim = 1;
    tilingparams.block_tiling_axis = 0;
    tilingparams.block_factor = input_x[0];
    // judge exceed ub_size and do workspace
    bool isworkspace = CheckWorkspaceCase(input_x, reduce_axis, max_ub_size);
    bool isexceedub = CheckExceedUbSize(input_x[0], input_x, 0, max_ub_size);
    if (isworkspace && isexceedub) {
      // workspace
      // not open and open multi-core block
      std::vector<int32_t> ubtilingdata = GetUbTilingData(
          input_x[0], 0, input_x, max_ub_size, reduce_axis, block);
      tilingparams.ub_tiling_axis = ubtilingdata[0];
      tilingparams.ub_factor = ubtilingdata[1];
      int32_t ub_tiling_axis_reduce =
          GetUbReduceAxis(input_x, max_ub_size, reduce_axis);
      tilingparams.ub_tiling_axis_reduce = ub_tiling_axis_reduce;
      tilingparams.ub_fuse_factor = input_x[0];
      compileinfo.is_normal = false;
    } else {
      // normal case
      tilingparams.ub_tiling_axis = 0;
      tilingparams.ub_factor = input_x[0];
      tilingparams.ub_tiling_axis_reduce = 0;
      tilingparams.ub_fuse_factor = 0;
      compileinfo.is_normal = true;
    }
  }
  for (size_t i = 0; i < input_x.size(); i++) {
    if (IsInVector(reduce_axis, i)) {
      continue;
    } else {
      if (input_x[i] > core_num) {
        int32_t mul_num = GetUnblockAxisOutputMul(i, input_x, reduce_axis);
        int32_t block_inner_core = input_x[i] % core_num == 0
                                       ? input_x[i] / core_num
                                       : (input_x[i] + core_num + 1) / core_num;
        int32_t min_block_inner_core = block % mul_num == 0
                                           ? block / mul_num
                                           : (block + mul_num) / mul_num;
        if (block_inner_core < min_block_inner_core) {
          block_inner_core = min_block_inner_core;
        }

        int32_t block_inner = block_inner_core;

        tilingparams.block_dim =
            input_x[i] % block_inner == 0
                ? input_x[i] / block_inner
                : (input_x[i] + block_inner + 1) / block_inner;
        tilingparams.block_tiling_axis = i;
        tilingparams.block_factor = block_inner;

        // judge exceed ub_size and do workspace
        bool isworkspace =
            CheckWorkspaceCase(input_x, reduce_axis, max_ub_size);
        bool isexceedub =
            CheckExceedUbSize(block_inner, input_x, i, max_ub_size);

        if (isworkspace && isexceedub) {
          // workspace

          // not open and open multi-core block
          std::vector<int32_t> ubtilingdata = GetUbTilingData(
              block_inner, i, input_x, max_ub_size, reduce_axis, block);
          tilingparams.ub_tiling_axis = ubtilingdata[0];
          tilingparams.ub_factor = ubtilingdata[1];
          int32_t ub_tiling_axis_reduce =
              GetUbReduceAxis(input_x, max_ub_size, reduce_axis);

          tilingparams.ub_tiling_axis_reduce = ub_tiling_axis_reduce;

          /***
           * ub_fuse_factor condition:
           * open multi-core block
           * block_inner < input_x[0]                  /
           * ub_fuse_factor > block                    /
           * block_inner > block                       / --> if true:
           *ub_fuse_factor = max([block,、、、， block_inner]) block_inner %
           *ub_fuse_factor > block      /      else:ub_fuse_factor = block_inner
           * ub_fuse_factor < ub_size                  /
           *
           * not open:
           * ub_fuse_factor = 1
           ***/

          if (block_inner < input_x[0] && block_inner > block) {
            for (int32_t n = block_inner; n > block; n--) {
              if (n < max_ub_size && block_inner % n > block) {
                tilingparams.ub_fuse_factor = n;
                break;
              }
            }
          } else {
            tilingparams.ub_fuse_factor = block_inner;
          }
          compileinfo.is_normal = false;
        } else if (isexceedub && !isworkspace) {
          // open multi-core block and split ub axis
          std::vector<int32_t> ubtilingdata = GetUbTilingData(
              block_inner, i, input_x, max_ub_size, reduce_axis, block);
          tilingparams.ub_tiling_axis = ubtilingdata[0];
          tilingparams.ub_factor = ubtilingdata[1];
          bool normal2workspace = ubtilingdata[2];
          if (normal2workspace) {
            tilingparams.ub_tiling_axis_reduce = 0;
            tilingparams.ub_fuse_factor = block_inner;
            compileinfo.is_normal = false;
          } else {
            bool is_open_multi_core = ubtilingdata[3];
            if (!is_open_multi_core) {
              tilingparams.block_dim = 1;
              tilingparams.block_tiling_axis = i;
              tilingparams.block_tiling_axis_1 = i;
              tilingparams.block_factor_1 = 1;
              tilingparams.block_factor = input_x[i];
            } else if (tilingparams.ub_tiling_axis != i) {
              std::vector<int32_t> blocktilingdata = GetBlockTilingData(
                  i, block_inner, ubtilingdata[0], ubtilingdata[4], input_x,
                  reduce_axis, core_num);
              tilingparams.block_dim = blocktilingdata[1] * blocktilingdata[3];
              tilingparams.block_tiling_axis = i;
              tilingparams.block_tiling_axis_1 = blocktilingdata[2];
              tilingparams.block_factor = blocktilingdata[1];
              tilingparams.block_factor_1 = blocktilingdata[3];
            }
            tilingparams.ub_tiling_axis_reduce = 0;
            tilingparams.ub_fuse_factor = 0;
            compileinfo.is_normal = true;
          }
        } else {
          // normal case
          tilingparams.ub_tiling_axis = i;
          tilingparams.ub_factor = block_inner;
          tilingparams.ub_tiling_axis_reduce = 0;
          tilingparams.ub_fuse_factor = 0;
          compileinfo.is_normal = true;
        }
        break;
      } else {
        int32_t mul_num = GetUnblockAxisOutputMul(i, input_x, reduce_axis);
        int32_t block_inner_core = 1;
        int32_t min_block_inner_core = block % mul_num == 0
                                           ? block / mul_num
                                           : (block + mul_num) / mul_num;
        if (block_inner_core < min_block_inner_core) {
          block_inner_core = min_block_inner_core;
        }

        int32_t block_inner = block_inner_core;

        tilingparams.block_tiling_axis = i;
        tilingparams.block_factor = block_inner;
        tilingparams.block_dim =
            input_x[i] % block_inner == 0
                ? input_x[i] / block_inner
                : (input_x[i] + block_inner + 1) / block_inner;

        // judge exceed ub_size and do workspace
        bool isworkspace =
            CheckWorkspaceCase(input_x, reduce_axis, max_ub_size);
        bool isexceedub =
            CheckExceedUbSize(block_inner, input_x, i, max_ub_size);

        if (isworkspace && isexceedub) {
          // workspace
          // not open and open multi-core block
          std::vector<int32_t> ubtilingdata = GetUbTilingData(
              block, i, input_x, max_ub_size, reduce_axis, block);
          tilingparams.ub_tiling_axis = ubtilingdata[0];
          tilingparams.ub_factor = ubtilingdata[1];
          int32_t ub_tiling_axis_reduce =
              GetUbReduceAxis(input_x, max_ub_size, reduce_axis);
          tilingparams.ub_tiling_axis_reduce = ub_tiling_axis_reduce;
          tilingparams.ub_fuse_factor = block_inner;
          compileinfo.is_normal = false;
        } else if (isexceedub && !isworkspace) {
          // open multi-core block and split ub axis
          std::vector<int32_t> ubtilingdata = GetUbTilingData(
              block, i, input_x, max_ub_size, reduce_axis, block);
          tilingparams.ub_tiling_axis = ubtilingdata[0];
          tilingparams.ub_factor = ubtilingdata[1];
          bool normal2workspace = ubtilingdata[2];
          if (normal2workspace) {
            tilingparams.ub_tiling_axis_reduce = 0;
            tilingparams.ub_fuse_factor = block_inner;
            compileinfo.is_normal = false;
          } else {
            bool is_open_multi_core = ubtilingdata[3];
            if (!is_open_multi_core) {
              tilingparams.block_dim = 1;
              tilingparams.block_tiling_axis = i;
              tilingparams.block_tiling_axis_1 = i;
              tilingparams.block_factor_1 = 1;
              tilingparams.block_factor = input_x[i];
            } else if (tilingparams.ub_tiling_axis != i) {
              std::vector<int32_t> blocktilingdata = GetBlockTilingData(
                  i, block_inner, ubtilingdata[0], ubtilingdata[4], input_x,
                  reduce_axis, core_num);
              tilingparams.block_dim = blocktilingdata[1] * blocktilingdata[3];
              tilingparams.block_tiling_axis = i;
              tilingparams.block_tiling_axis_1 = blocktilingdata[2];
              tilingparams.block_factor = blocktilingdata[1];
              tilingparams.block_factor_1 = blocktilingdata[3];
            }
            tilingparams.ub_tiling_axis_reduce = 0;
            tilingparams.ub_fuse_factor = 0;
            compileinfo.is_normal = true;
          }
        } else {
          // normal case
          tilingparams.ub_tiling_axis = i;
          tilingparams.ub_factor = block_inner;
          tilingparams.ub_tiling_axis_reduce = 0;
          tilingparams.ub_fuse_factor = 0;
          compileinfo.is_normal = true;
        }
        break;
      }
    }
  }
}

bool LayerNormTiling(const std::string &op_type, const TeOpParas &op_paras,
                     const nlohmann::json &op_info, OpRunInfo &run_info) {
  OP_LOGI(op_type.c_str(), "LayerNormTiling running.");
  std::vector<int64_t> input_x = op_paras.inputs[0].tensor[0].shape;
  const std::string input_dtype = op_paras.inputs[0].tensor[0].dtype;

  std::vector<int64_t> input_gama = op_paras.inputs[1].tensor[0].shape;
  std::vector<int64_t> input_beta = op_paras.inputs[2].tensor[0].shape;
  std::vector<int32_t> reduce_axis =
      op_info["reduce_axis"].get<std::vector<int32_t>>();
  int32_t core_num = op_info["core_num"].get<int32_t>();

  int32_t max_ub_size;
  if (input_dtype == "float32" || input_dtype == "fp32") {
    max_ub_size = op_info["max_ub_size_normal_fp32"].get<int32_t>();
  } else {
    max_ub_size = op_info["max_ub_size_normal_fp16"].get<int32_t>();
  }
  int32_t workspace_sub1 = 4;
  for (uint32_t i = 0; i < input_x.size(); i++) {
    workspace_sub1 *= input_x[i];
    ByteBufferPut(run_info.tiling_data, (int32_t)input_x[i]);
    OP_LOGD(op_type.c_str(), "input_x shape:%d.", input_x[i]);
  }

  std::vector<int64_t> workspaces = {workspace_sub1};
  TilingParams tilingparams;
  CompileInfo compileinfo;
  bool compileflag = GetCompileInfo(op_type, op_info, compileinfo, reduce_axis,
                                    input_x, run_info);

  if (!compileflag) {
    OP_LOGE("op[%s] GetCompileInfo failed.", op_type.c_str());
  }

  GetTilingData(input_x, tilingparams, reduce_axis, core_num, max_ub_size,
                compileinfo, input_dtype);

  // tiling_key
  int32_t tiling_key =
      CalcTilingKey(compileinfo, input_x, tilingparams, reduce_axis);

  run_info.workspaces = workspaces;
  run_info.block_dim = tilingparams.block_dim;
  run_info.tiling_key = tiling_key;

  ByteBufferPut(run_info.tiling_data, (int32_t)tilingparams.block_factor);
  ByteBufferPut(run_info.tiling_data, (int32_t)tilingparams.block_factor_1);
  ByteBufferPut(run_info.tiling_data, (int32_t)tilingparams.ub_factor);
  ByteBufferPut(run_info.tiling_data, (int32_t)tilingparams.ub_fuse_factor);

  OP_LOGI(op_type.c_str(), "LayerNormTiling end.");
  return true;
}

// register tiling interface of LayerNorm op.
REGISTER_OP_TILING_FUNC_BUFFERED(LayerNorm, LayerNormTiling);
} // namespace optiling