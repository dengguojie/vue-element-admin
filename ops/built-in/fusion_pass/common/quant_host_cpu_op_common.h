/**
 * @file quant_common.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief fe_util
 *
 * @version 1.0
 *
 */
#ifndef BUILTIN_FUSIONPASS_QUANT_COMMON_H
#define BUILTIN_FUSIONPASS_QUANT_COMMON_H
#include "external/graph/types.h"
#include "graph/compute_graph.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include <string>
#include <vector>

namespace fe {
const int32_t INDEX_CO = 0;
const int32_t INDEX_CI = 1;
const int32_t INDEX_FILTER_H = 2;
const int32_t INDEX_FILTER_W = 3;
const int32_t LAST_AXIS_INDEX = 3;

/* Attribute Name */
static const std::string QUANT_SCALE = "quant_scale";
static const std::string DEQUANT_SCALE = "dequant_scale";
static const std::string ATTR_OFFSET_X = "offset_x";
static const std::string ATTR_OFFSET_W = "offset_w";
static const std::string ATTR_OFFSET = "offset";
static const std::string ATTR_SCALE = "scale";

fe::Status GetkernelDataCountForPass(const std::vector<int64_t> &filterDIms,
                                     int64_t &kernelDataCount);
} // namespace fe
#endif // BUILTIN_FUSIONPASS_QUANT_COMMON_H