/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include "scope_batchmulticlass2_nms_pass.h"
#include "scope_decode_bbox_v2_pass.h"
#include "scope_normalize_bbox_pass.h"
#include "scope_to_absolute_bbox_pass.h"

namespace ge {
REGISTER_SCOPE_FUSION_PASS("ScopeDynamicLSTMPass", ScopeDynamicLSTMPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeBasicLSTMCellPass", ScopeBasicLSTMCellPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeLayerNormPass", ScopeLayerNormPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeLayerNormGradPass", ScopeLayerNormGradPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeClipBoxesPass", ScopeClipBoxesPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeROIAlignPass", ScopeROIAlignPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeRpnProposalsPass", ScopeRpnProposalsPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeFastrcnnPredictionsPass", ScopeFastrcnnPredictionsPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeDecodeBboxPass", ScopeDecodeBboxPass, true);
REGISTER_SCOPE_FUSION_PASS("ScopeBatchMultiClassNonMaxSuppressionPass", ScopeBatchMultiClassNonMaxSuppressionPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeBatchMultiClassNMSPass", ScopeBatchMultiClassNMSPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeDecodeBboxV2Pass", ScopeDecodeBboxV2Pass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeNormalizeBBoxPass", ScopeNormalizeBBoxPass, false);
REGISTER_SCOPE_FUSION_PASS("ScopeToAbsoluteBBoxPass", ScopeToAbsoluteBBoxPass, false);

}  // namespace ge
