/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_
#define FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_

#include <string>
#include "graph/types.h"

namespace fe {
static const std::string CORE_TYPE = "_coretype";
/* engine name of AI core and vector core */
static const std::string AI_CORE_NAME = "AIcoreEngine";
static const std::string VECTOR_CORE_NAME = "VectorEngine";

static const std::string L1_OPTIMIZE = "l1_optimize";
static const std::string L2_OPTIMIZE = "l2_optimize";
static const std::string OFF_OPTIMIZE = "off_optimize";

/* allow auto mix precision */
static const std::string ALLOW_MIX_PRECISION = "allow_mix_precision";

/* force float16  */
static const std::string FORCE_FP16 = "force_fp16";


/* force float32  */
static const std::string FORCE_FP32 = "force_fp32";

/* allow fp32 to fp16 */
static const std::string ALLOW_FP32_TO_FP16 = "allow_fp32_to_fp16";

/* must remain origin dtype */
static const std::string MUST_KEEP_ORIGIN_DTYPE = "must_keep_origin_dtype";

static const int64_t IS_UNKNOWN_SHAPE_VALUE = 1;

static const int64_t SHAPE_UNKNOWN_DIM = -1;

static const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;

static const uint32_t INVALID_DTYPE_AND_FORMAT_SIZE = 0xffffffff;

static const std::string SOC_VERSION_ASCEND310 = "Ascend310";
static const std::string SOC_VERSION_ASCEND610 = "Ascend610";
static const std::string SOC_VERSION_ASCEND615 = "Ascend615";
static const std::string SOC_VERSION_ASCEND710 = "Ascend710";
static const std::string SOC_VERSION_ASCEND710P = "Ascend710Pro";
static const std::string SOC_VERSION_ASCEND910 = "Ascend910";
static const std::string SOC_VERSION_ASCEND910A = "Ascend910A";
static const std::string SOC_VERSION_ASCEND910B = "Ascend910B";
static const std::string SOC_VERSION_ASCEND910PROA = "Ascend910ProA";
static const std::string SOC_VERSION_ASCEND910PROB = "Ascend910ProB";
static const std::string SOC_VERSION_ASCEND920A = "Ascend920A";
static const std::string SOC_VERSION_ASCEND910PREMIUMA = "Ascend910PremiumA";
static const std::string SOC_VERSION_HI3796CV300ES = "Hi3796CV300ES";
static const std::string SOC_VERSION_HI3796CV300CS = "Hi3796CV300CS";
static const std::string SOC_VERSION_SD3403 = "SD3403";

static const std::vector<std::string> SOC_VERSION_CLOUD_LIST = {
        SOC_VERSION_ASCEND910A, SOC_VERSION_ASCEND910B, SOC_VERSION_ASCEND910PROA,
        SOC_VERSION_ASCEND910PROB, SOC_VERSION_ASCEND910PREMIUMA
};

static const std::vector<std::string> SOC_VERSION_DC_LIST = {SOC_VERSION_ASCEND710, SOC_VERSION_ASCEND710P};

static const std::vector<std::string> SOC_VERSION_MDC_LIST = {SOC_VERSION_ASCEND610, SOC_VERSION_ASCEND615};

static const std::vector<std::string> SOC_VERSION_MDCOrDC_LIST = {SOC_VERSION_ASCEND610, SOC_VERSION_ASCEND615,
                                                                  SOC_VERSION_ASCEND710, SOC_VERSION_ASCEND710P};

static const std::map<ge::DataType, uint32_t> DATATYPE_SIZE_MAP {
        {ge::DT_FLOAT, sizeof(float)},
        {ge::DT_FLOAT16, sizeof(int16_t)},
        {ge::DT_INT8, sizeof(int8_t)},
        {ge::DT_INT32, sizeof(int32_t)},
        {ge::DT_UINT8, sizeof(uint8_t)},
        {ge::DT_UINT32, sizeof(uint32_t)},
        {ge::DT_INT16, sizeof(int16_t)},
        {ge::DT_UINT16, sizeof(uint16_t)},
        {ge::DT_INT64, sizeof(int64_t)},
        {ge::DT_UINT64, sizeof(uint64_t)},
        {ge::DT_DOUBLE, sizeof(double)},
        {ge::DT_BOOL, sizeof(bool)},
        {ge::DT_DUAL, sizeof(float) + sizeof(int8_t)},
        {ge::DT_DUAL_SUB_UINT8, sizeof(int8_t)},
        {ge::DT_DUAL_SUB_INT8, sizeof(int8_t)}
};
static  const std::map<std::string, std::string> LICENSE_PASSNAME_MAP {
  {"1", "ConstToAttrPass"},
  {"2", "ConstToAttrReduceSumFusion"},
  {"3", "MatMulBiasAddFusionPass"},
  {"4", "OneHotFusionPass"},
  {"5", "TileConstToAttrFusion"},
  {"6", "MulAddNL2LossFusionPass"},
  {"7", "AutomaticUbFusion"},
  {"8", "DreluFusionPass"},
  {"9", "TransposeReshapeFusionPass"},
  {"10", "A_MomentumLossscaleFusionPass"},
  {"11", "ApplyAddOutputPass"},
  {"12", "FusedBatchnormFusionPass"},
  {"13", "FusedBatchNormGradFusionPass"},
  {"14", "MaxPoolWithArgmaxFusionPass"},
  {"15", "TbeBnupdateEltwiseEltwiseFusionPass"},
  {"16", "TbeBnupdateEltwiseFusionPass"},
  {"17", "TbeConv2DBackpropElemwiseFusionPass"},
  {"18", "TbeConvBnreduceFusionPass"},
  {"19", "BatchMatmulFusionPass"},
  {"20", "ConstToAttrStridedSliceFusion"},
  {"21", "ExtremumGradFusionPass"},
  {"22", "LayerNormGradFusionPass"},
  {"23", "LayerNormGradFusionPassBetaGammaV2"},
  {"24", "LogSoftmaxGradFusionPass"},
  {"25", "MatmulCastFusionPass"},
  {"26", "ReshapeTransposeFusionPass"},
  {"27", "SquareSumV1"},
  {"28", "StridedSliceGradFusionPass"},
  {"29", "ZUnsortedSegmentSumUpdateFusionPass"},
  {"30", "ATbeMatmulElemwiseFusionPass"},
  {"31", "BatchMatmulConfusiontransposeUbFusion"},
  {"32", "MatmulConfusiontransposeUbFusion"},
  {"33", "TbeBatchMatmulFusedMulAddFusionPass"},
  {"34", "TbeEltwiseFusionPass"},
  {"35", "TbeFullyconnectionElemwiseDequantFusionPass"},
  {"36", "TbeMultiOutputFusionPass"},
  {"37", "MulAddFusionPass"},
  {"38", "SoftmaxGradExtFusion"},
  {"39", "clip_by_norm_nodivsquaresum"}
};

static const std::vector<ge::Format> FE_ORIGIN_FORMAT_VECTOR = {ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
                                                                ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
                                                                ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND};

static const std::vector<ge::Format> FE_HEAVY_FORMAT_VECTOR = {
        ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NC1HWC0,  ge::FORMAT_C1HWNCoC0,    ge::FORMAT_FRACTAL_Z,
        ge::FORMAT_FRACTAL_NZ,  ge::FORMAT_NDC1HWC0, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_
