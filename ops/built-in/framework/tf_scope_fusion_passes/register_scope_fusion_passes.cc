/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file register_scope_fusion_passes.cc
 * \brief
 */
#include "register/scope/scope_fusion_pass_register.h"

#include "scope_dynamic_lstm_pass.h"
#include "scope_basic_lstm_cell_pass.h"
#include "scope_layernorm_pass.h"
#include "scope_layernorm_grad_pass.h"
#include "scope_clip_boxes_pass.h"
#include "scope_roi_align_pass.h"
#include "scope_rpn_proposals_pass.h"
#include "scope_fastrcnn_predictions_pass.h"
#include "scope_decode_bbox_pass.h"
#include "scope_batchmulticlass_nms_pass.h"
#include "scope_newbatchmulticlass_nms_pass.h"
#include "scope_decode_bbox_v2_pass.h"
#include "scope_normalize_bbox_pass.h"
#include "scope_to_absolute_bbox_pass.h"
#include "scope_preprocess_keep_ratio_resize_bilinear_pass.h"
#include "scope_dynamic_rnn_pass.h"
#include "scope_dynamic_gru_pass.h"
#include "scope_instancenorm_pass.h"
#include "scope_instancenorm_grad_pass.h"

namespace ge {
REGISTER_SCOPE_FUSION_PASS("ScopeBasicLSTMCellPass", ScopeBasicLSTMCellPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeLayerNormPass", ScopeLayerNormPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeLayerNormGradPass", ScopeLayerNormGradPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeInstanceNormPass", ScopeInstanceNormPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeInstanceNormGradPass", ScopeInstanceNormGradPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeClipBoxesPass", ScopeClipBoxesPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeROIAlignPass", ScopeROIAlignPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeRpnProposalsPass", ScopeRpnProposalsPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeFastrcnnPredictionsPass", ScopeFastrcnnPredictionsPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeDecodeBboxPass", ScopeDecodeBboxPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeKeepRatioResizeBilinearPass", ScopeKeepRatioResizeBilinearPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeBatchMultiClassNonMaxSuppressionPass", ScopeBatchMultiClassNonMaxSuppressionPass,
                           false);
REGISTER_SCOPE_FUSION_PASS("ScopeBatchMultiClassNMSPass", ScopeBatchMultiClassNMSPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeDecodeBboxV2Pass", ScopeDecodeBboxV2Pass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeNormalizeBBoxPass", ScopeNormalizeBBoxPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeToAbsoluteBBoxPass", ScopeToAbsoluteBBoxPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeDynamicRNNPass", ScopeDynamicRNNPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeDynamicGRUPass", ScopeDynamicGRUPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeDynamicLSTMPass", ScopeDynamicLSTMPass, false);

}  // namespace ge
