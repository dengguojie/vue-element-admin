/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file conv2d_slice_info_cal_base.h
 * \brief tbe conv2d slice info cal
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_COMMON_CONV2D_SLICE_INFO_CAL_BASE_H
#define OPS_BUILT_IN_FUSION_PASS_COMMON_CONV2D_SLICE_INFO_CAL_BASE_H

#include <vector>
#include <map>
#include "common/lxfusion_json_util.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class ConvSliceInfoCalBase {
public:
    explicit ConvSliceInfoCalBase();
    virtual ~ConvSliceInfoCalBase();
    // Calc slice info main function
    Status ConvCalcFusionOpSliceInfo(std::vector<ge::NodePtr> &fusion_nodes,
                                     OpCalcInfo &op_slice_info, std::string fused_op_type);

    #define CONV_RET_IF_SMART_PTR_IS_NULL(smart_ptr) do {                                \
        if (!(smart_ptr)) {                                                              \
            OP_LOGE(fused_op_type_.c_str(), "new an object failed.");                    \
            return FAILED;                                                               \
        }                                                                                \
    } while (0)
protected:
private:
    bool is_head_fusion_ = false; // head or tail fusion flag
    const int64_t INPUT_BATCH_AXIS = 0;
    const int64_t OUTPUT_COUT_AXIS = 1;
    const int64_t INPUT_H_AXIS = 2;
    const int64_t INPUT_W_AXIS = 3;
    const uint CONV_FIRST_INPUT_IDX = 0;
    const uint CONV_SECOND_INPUT_IDX = 1;
    const std::string slice_info_cal_type_ = "Conv2DSliceInfoBaseCal";
    std::string fused_op_type_ = "";
    std::string SLICE_CAL_ELTWISE = "Eltwise";
    std::string SLICE_CAL_LEAKYRELU = "LeakyRelu";
    std::string SLICE_CAL_VADD = "Vadd";
    std::string SLICE_CAL_RELU = "Relu";
    std::string SLICE_CAL_ADD = "Add";
    std::string SLICE_CAL_RELU6 = "Relu6";
    std::string SLICE_CAL_Mul = "Mul";
    const std::vector<std::string> ELEM_WISE_WHITE_LIST = {SLICE_CAL_ELTWISE, SLICE_CAL_LEAKYRELU, SLICE_CAL_ADD,\
                                                           SLICE_CAL_VADD, SLICE_CAL_RELU, SLICE_CAL_RELU6, SLICE_CAL_Mul};
    // Calc slice info del/update function
    void ConvDelSplitInfoByOpType(std::vector<ge::NodePtr> &fusion_nodes, std::vector<AxisSplitMap> &split_maps);
    void ConvUpdateSplitInfoByOpType(std::vector<ge::NodePtr> &fusion_nodes,
                                     std::vector<AxisSplitMap> &split_maps, int *input_base_idx, int *output_base_idx);
    // conv split info del functions
    typedef void (ConvSliceInfoCalBase::*convSliceInfoDel)(std::vector<AxisSplitMap> &split_maps);
    void BNReduceSplitInfoDel(std::vector<AxisSplitMap> &split_maps);
    void QuantAndRequantSplitInfoDel(std::vector<AxisSplitMap> &split_maps);
    void StrideWriteSplitInfoDel(std::vector<AxisSplitMap> &split_maps);
    std::map<std::string, convSliceInfoDel> opDelFunMap = {
        {"BNTrainingReduce", &ConvSliceInfoCalBase::BNReduceSplitInfoDel},
        {"AscendQuant", &ConvSliceInfoCalBase::QuantAndRequantSplitInfoDel},
        {"AscendRequant", &ConvSliceInfoCalBase::QuantAndRequantSplitInfoDel},
        {"AscendRequantS16", &ConvSliceInfoCalBase::QuantAndRequantSplitInfoDel},
        {"StridedWrite", &ConvSliceInfoCalBase::StrideWriteSplitInfoDel},
    };

    // conv split info input update functions
    typedef void (ConvSliceInfoCalBase::*convSliceInfoUpdate)(std::vector<AxisSplitMap> &split_maps, int *idx_base);
    void DequantInputUpdate(std::vector<AxisSplitMap> &split_maps, int *input_idx_base);
    void DequantS16InputUpdate(std::vector<AxisSplitMap> &split_maps, int *input_idx_base);
    void RequantS16InputUpdate(std::vector<AxisSplitMap> &split_maps, int *input_idx_base);
    void ElemwiseInputUpdate(std::vector<AxisSplitMap> &split_maps, int *input_idx_base);
    std::map<std::string, convSliceInfoUpdate> opInputUpdateFunMap = {
        {"AscendDequant", &ConvSliceInfoCalBase::DequantInputUpdate},
        {"AscendDequantS16", &ConvSliceInfoCalBase::DequantS16InputUpdate},
        {"AscendRequantS16", &ConvSliceInfoCalBase::RequantS16InputUpdate},
        {"Elemwise", &ConvSliceInfoCalBase::ElemwiseInputUpdate},
    };

    // conv split info output update functions
    void BnreduceOutputUpdate(std::vector<AxisSplitMap> &split_maps, int *output_idx_base);
    void UtilOutputUpdate(std::vector<AxisSplitMap> &split_maps, int *output_idx_base);
    std::map<std::string, convSliceInfoUpdate> opOutputUpdateFunMap = {
        {"BNTrainingReduce", &ConvSliceInfoCalBase::BnreduceOutputUpdate},
        {"ReluV2", &ConvSliceInfoCalBase::UtilOutputUpdate},
        {"AscendRequantS16", &ConvSliceInfoCalBase::UtilOutputUpdate},
        {"Elemwise", &ConvSliceInfoCalBase::UtilOutputUpdate},
    };

    // util functions
    void ConvDelSplitInfoByInputIdxAndAxis(std::vector<AxisSplitMap>& split_maps, uint idx, int axis);
    bool IsHeadFusion(const ge::NodePtr &fusion_node, const std::vector<ge::NodePtr> &fusion_nodes);
    bool GetOpSliceInfoL1Fusion(OpL1FusionType& fusion_enable, ge::NodePtr& cube_node, const string& fused_op_type);
};
} // namespace fe

#endif // OPS_BUILT_IN_FUSION_PASS_COMMON_CONV2D_SLICE_INFO_CAL_BASE_H