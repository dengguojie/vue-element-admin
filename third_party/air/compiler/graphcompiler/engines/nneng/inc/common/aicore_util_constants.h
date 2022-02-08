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
#include <map>
#include <vector>
#include <unordered_set>
#include "graph/types.h"

namespace fe {
/* engine name of AI core and vector core */
const std::string AI_CORE_NAME = "AIcoreEngine";
const std::string VECTOR_CORE_NAME = "VectorEngine";

const std::string L1_OPTIMIZE = "l1_optimize";
const std::string L2_OPTIMIZE = "l2_optimize";
const std::string OFF_OPTIMIZE = "off_optimize";

/* allow auto mix precision */
const std::string ALLOW_MIX_PRECISION = "allow_mix_precision";
const std::string ALLOW_MIX_PRECISION_FP16 = "allow_mix_precision_fp16";
const std::string ALLOW_MIX_PRECISION_BF16 = "allow_mix_precision_bf16";

/* force float16  */
const std::string FORCE_FP16 = "force_fp16";
/* force bf16  */
const std::string FORCE_BF16 = "force_bf16";
/* force lowerprecison  */
const std::string FORCE_LOWERPRECISION = "force_lowerprecision";


/* force float32  */
const std::string FORCE_FP32 = "force_fp32";

/* allow fp32 to fp16 */
const std::string ALLOW_FP32_TO_FP16 = "allow_fp32_to_fp16";
/* allow fp32 to bf16 */
const std::string ALLOW_FP32_TO_BF16 = "allow_fp32_to_bf16";
/* allow fp32 to lowerprecision */
const std::string ALLOW_FP32_TO_LOWPRECISION = "allow_fp32_to_lowprecision";
/* need to update dtype when op checksupport*/
const std::string NEED_UPDATE_DTYPE_WHEN_OP_CHECKSUPPORT = "need_update_dtype_when_op_checksupport";

/* must remain origin dtype */
const std::string MUST_KEEP_ORIGIN_DTYPE = "must_keep_origin_dtype";

const std::string SHAPE_GENERALIZED = "shape_generalized";

const int64_t IS_UNKNOWN_SHAPE_VALUE = 1;

// mode == 1 indicates we need reserve 8 Bytes for the args beginning
const int64_t IS_MIX_FIRST_FIELD_MODE = 1;

const int64_t SHAPE_UNKNOWN_DIM = -1;

const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;

const uint32_t INVALID_DTYPE_AND_FORMAT_SIZE = 0xffffffff;

const size_t kMaxPretchNum = 4;

const size_t kMinThreadNum = 2;

const int32_t kAnchorMaxIdxNum = 64;

const size_t kMaxOpNmaLen = 512;

const std::string kCoreTypeAIC = "AIC";

const std::string kCoreTypeAIV = "AIV";

const std::string kCoreTypeMixAIC = "MIX_AIC";

const std::string kCoreTypeMixAIV = "MIX_AIV";

const std::string kMixL2CtxDef = "mix_l2_ctx_def";

const std::string kCoreTypeHcomReduce = "HcomReduce";
const std::string kCoreTypeHcomAllReduce = "HcomAllReduce";
const std::string kCoreTypeHcomAllGather = "HcomAllGather";
const std::string kCoreTypeHcomBroadCast = "HcomBroadcast";
const std::string kCoreTypeHcomReduceScatter = "HcomReduceScatter";
const std::string kCoreTypeHcomSend = "HcomSend";
const std::string kCoreTypeHcomReceive = "HcomReceive";
const std::string kCoreTypeHcomRemoteRead = "HcomRemoteRead";
const std::string kCoreTypeHcomRemoteRefRead = "HcomRemoteRefRead";
const std::string kCoreTypeHcomRemoteWrite = "HcomRemoteWrite";
const std::string kCoreTypeHcomRemoteScatterWrite= "HcomRemoteScatterWrite";
const std::string kCoreTypeHcomAllToAllV = "HcomAllToAllV";
const std::string kCoreTypeHcomGatherAllToAllV = "HcomGatherAllToAllV";
const std::unordered_set<std::string> kHCCLOpType = {kCoreTypeHcomReduce, kCoreTypeHcomAllReduce, kCoreTypeHcomAllGather,
                                                     kCoreTypeHcomBroadCast, kCoreTypeHcomReduceScatter, kCoreTypeHcomSend,
                                                     kCoreTypeHcomReceive, kCoreTypeHcomRemoteRead, kCoreTypeHcomRemoteRefRead,
                                                     kCoreTypeHcomRemoteWrite, kCoreTypeHcomRemoteScatterWrite, kCoreTypeHcomAllToAllV,
                                                     kCoreTypeHcomGatherAllToAllV};

const std::string kKernelName = "_kernelname";

const std::string kThreadKernelName = "_thread_kernelname";

const std::string kTypePhonyConcat = "PhonyConcat";

const std::string kAtomicAddrClean = "AtomicAddrClean";

const uint32_t kMaxIdx = 64;

const uint32_t kGetFirstAvailableLabel = 1000;

const std::string kFFTSPlusMode = "ffts-plus";

const std::string kFFTSMode = "ffts";

const std::string kAICoreMode = "ffts_mode";

const std::string kSocInfo = "SoCInfo";

const std::string SOC_VERSION_ASCEND310 = "Ascend310";
const std::string SOC_VERSION_ASCEND610 = "Ascend610";
const std::string SOC_VERSION_ASCEND615 = "Ascend615";
const std::string SOC_VERSION_ASCEND710 = "Ascend710";
const std::string SOC_VERSION_ASCEND710P = "Ascend710Pro";
const std::string SOC_VERSION_ASCEND910 = "Ascend910";
const std::string SOC_VERSION_ASCEND910A = "Ascend910A";
const std::string SOC_VERSION_ASCEND910B = "Ascend910B";
const std::string SOC_VERSION_ASCEND910PROA = "Ascend910ProA";
const std::string SOC_VERSION_ASCEND910PROB = "Ascend910ProB";
const std::string SOC_VERSION_ASCEND920A = "Ascend920A";
const std::string SOC_VERSION_ASCEND320 = "Ascend320";
const std::string SOC_VERSION_ASCEND910PREMIUMA = "Ascend910PremiumA";
const std::string SOC_VERSION_HI3796CV300ES = "Hi3796CV300ES";
const std::string SOC_VERSION_HI3796CV300CS = "Hi3796CV300CS";
const std::string SOC_VERSION_SD3403 = "SD3403";

const std::string kStrTrue = "true";
const std::string kStrFalse = "false";

const std::vector<std::string> SOC_VERSION_CLOUD_LIST = {
        SOC_VERSION_ASCEND910A, SOC_VERSION_ASCEND910B, SOC_VERSION_ASCEND910PROA,
        SOC_VERSION_ASCEND910PROB, SOC_VERSION_ASCEND910PREMIUMA
};

const std::vector<std::string> SOC_VERSION_DC_LIST = {SOC_VERSION_ASCEND710, SOC_VERSION_ASCEND710P};

const std::vector<std::string> SOC_VERSION_MDC_LIST = {SOC_VERSION_ASCEND610, SOC_VERSION_ASCEND615};

const std::vector<std::string> SOC_VERSION_MDCOrDC_LIST = {SOC_VERSION_ASCEND610, SOC_VERSION_ASCEND615,
                                                                  SOC_VERSION_ASCEND710, SOC_VERSION_ASCEND710P};

const std::vector<std::string> SOC_VERSION_LHISI_LIST = {SOC_VERSION_HI3796CV300ES, SOC_VERSION_HI3796CV300CS,
                                                                SOC_VERSION_SD3403};

enum CMO_Operation {
  BIT_MAP_OP = 0,
  CTX_TYPE_OP = 1,
  INPUT_FLAG_OP = 2,
  CMO_NAME_OP = 3
};

const std::map<ge::DataType, uint32_t> DATATYPE_SIZE_MAP {
        {ge::DT_FLOAT, sizeof(float)},
        {ge::DT_FLOAT16, sizeof(int16_t)},
        {ge::DT_BF16, sizeof(int16_t)},
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

const std::map<std::string, std::string> LICENSE_PASSNAME_MAP {
  {"3", "MatMulBiasAddFusionPass"},
  {"4", "OneHotFusionPass"},
  {"6", "MulAddNL2LossFusionPass"},
  {"7", "AutomaticUbFusion"},
  {"9", "TransposeReshapeFusionPass"},
  {"15", "TbeBnupdateEltwiseEltwiseFusionPass"},
  {"16", "TbeBnupdateEltwiseFusionPass"},
  {"17", "TbeConv2DBackpropElemwiseFusionPass"},
  {"18", "TbeConvBnreduceFusionPass"},
  {"19", "BatchMatmulFusionPass"},
  {"20", "ConstToAttrStridedSliceFusion"},
  {"21", "ExtremumGradFusionPass"},
  {"22", "LayerNormGradFusionPass"},
  {"23", "LayerNormGradFusionPassBetaGammaV2"},
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
  {"39", "clip_by_norm_nodivsquaresum"},
  {"40", "PadConv2dFusionPass"},
  {"41", "SparseSoftMaxFusionPass"},
  {"42", "MulAddNPass"},
  {"43", "Resnet50DbnDwFusionPass"},
  {"44", "BatchMatmulDropOutDoMaskV3DFusionPass"},
  {"45", "MatmulConfusiontransposeUbFusion"},
  {"46", "MatmulDropOutDoMaskV3DFusionPass"},
  {"47", "TbeBatchMatmulElementWiseFusionPass"},
  {"48", "SoftmaxWithDropOutDoMaskFusion"}
};

const std::vector<ge::Format> FE_ORIGIN_FORMAT_VECTOR = {ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
                                                                ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
                                                                ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND};

const std::vector<ge::Format> FE_HEAVY_FORMAT_VECTOR = {
        ge::FORMAT_NC1HWC0_C04, ge::FORMAT_NC1HWC0,  ge::FORMAT_C1HWNCoC0,    ge::FORMAT_FRACTAL_Z,
        ge::FORMAT_FRACTAL_NZ,  ge::FORMAT_NDC1HWC0, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_CONSTANTS_H_
