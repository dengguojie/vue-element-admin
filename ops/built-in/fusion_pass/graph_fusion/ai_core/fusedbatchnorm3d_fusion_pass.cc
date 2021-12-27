/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file fusedbatchnorm_fusion_pass.cpp
 * \brief fusedbatchnorm fusion pass
 *   (BatchNorm3D --> BN3DTrainingReduce & BN3DTrainingUpdate)
 */
#include "fusedbatchnorm3d_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include "op_log.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {
REGISTER_PASS("FusedBatchnorm3DFusionPass", BUILT_IN_GRAPH_PASS, FusedBatchnorm3DFusionPass);
}  // namespace fe
