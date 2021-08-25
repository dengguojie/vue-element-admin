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
 * \file conv2d_slice_info_cal_base.cc
 * \brief tbe conv2d slice info cal
 */
#include "conv2d_slice_info_cal_base.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "conv2d_slice_info_cal_base.h"
#include "lx_fusion_func.h"

using namespace std;

namespace fe {

ConvSliceInfoCalBase::ConvSliceInfoCalBase()
{
}

ConvSliceInfoCalBase::~ConvSliceInfoCalBase()
{
}

Status ConvSliceInfoCalBase::ConvCalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes,
    OpCalcInfo &op_slice_info, string fused_op_type)
{
    FUSION_PASS_CHECK(!fusion_nodes.size(), OP_LOGD("fusion node size is empty!"), return FAILED);
    auto first_fusion_node = fusion_nodes[0];
    FUSION_PASS_CHECK((first_fusion_node == nullptr), OP_LOGD("first fusion_node is null."), return FAILED);
    FUSION_PASS_CHECK(!strcmp(first_fusion_node->GetType().c_str(), "Conv2D") && \
                      !strcmp(first_fusion_node->GetType().c_str(), "Conv2DCompress") && \
                      !strcmp(first_fusion_node->GetType().c_str(), "DepthwiseConv2D"),
                      OP_LOGD("fusion node start is not conv node!"), return FAILED);
    OP_LOGD(fused_op_type.c_str(), "start calculate Conv2d fusion node slice info");
    fused_op_type_.assign(fused_op_type);
    vector<AxisSplitMap> split_maps;
    if(!GetSplitMap(split_maps, first_fusion_node, fused_op_type_)) {
        return FAILED;
    }
    int inputIdxBase = first_fusion_node->GetInDataNodes().size() - 1;
    int outputIdxBase = first_fusion_node->GetOutDataNodes().size() - 1;
    OP_LOGD(fused_op_type.c_str(),
            "start calculate Conv2d fusion node slice info in:%d, out:%d", inputIdxBase, outputIdxBase);
    ConvDelSplitInfoByOpType(fusion_nodes, split_maps);
    ConvUpdateSplitInfoByOpType(fusion_nodes, split_maps, &inputIdxBase, &outputIdxBase);
    op_slice_info.SetAxisSplitMaps(split_maps);
    OpL1FusionType fusion_enable;
    if (!GetOpSliceInfoL1Fusion(fusion_enable, first_fusion_node, fused_op_type_)) {
        return FAILED;
    }
    op_slice_info.SetL1FusionEnable(fusion_enable);
    int64_t convMinL1Space = 0;
    op_slice_info.SetMinTbeL1Space(convMinL1Space);
    OP_LOGD(fused_op_type.c_str(), "end Conv2d fusion node slice info");
    return SUCCESS;
}

void ConvSliceInfoCalBase::ConvDelSplitInfoByOpType(vector<ge::NodePtr> &fusion_nodes,
                                                      vector<AxisSplitMap> &split_maps)
{
    OP_LOGD("begin del Conv2d fusion node slice info");
    // op slice info del by op type
    for(auto &fusion_node: fusion_nodes) {
        string tmpOpType = fusion_node->GetType();
        auto iterDelFun = opDelFunMap.find(tmpOpType);
        if (iterDelFun != opDelFunMap.end()) {
            (this->*opDelFunMap[tmpOpType])(split_maps);
        } else {
            OP_LOGD(tmpOpType.c_str(), "op type del fun not support by now");
        }
    }
    OP_LOGD("end del Conv2d fusion node slice info");
    return;
}

void ConvSliceInfoCalBase::ConvUpdateSplitInfoByOpType(vector<ge::NodePtr> &fusion_nodes,
    vector<AxisSplitMap> &split_maps, int *inputBaseIdx, int *outputBaseIdx)
{
    OP_LOGD("begin update Conv2d fusion node slice info");
    // op slice info update by op type
    for (auto &fusion_node : fusion_nodes) {
        OP_LOGD(fused_op_type_.c_str(), "op fusion node is %s", fusion_node->GetType().c_str());
        string tmpOpType = fusion_node->GetType();
        if (find(ELEM_WISE_WHITE_LIST.begin(), ELEM_WISE_WHITE_LIST.end(), tmpOpType) != ELEM_WISE_WHITE_LIST.end()) {
            OP_LOGD(fusion_node->GetType().c_str(), "op fusion node rename elemwise");
            tmpOpType.assign("Elemwise");
        }
        // op input slice info update by op type
        auto iterInputUpdateFun = opInputUpdateFunMap.find(tmpOpType);
        if (iterInputUpdateFun != opInputUpdateFunMap.end()) {
            bool run_input_update_flag = true;
            // if dequant node's deq_scale input is not tensor, no need to add split info
            if (!strcmp(tmpOpType.c_str(), "AscendDequant") || !strcmp(tmpOpType.c_str(), "AscendDequantS16")) {
                auto deq_scale_tensor = fusion_node->GetOpDesc()->MutableInputDesc(1);
                auto deq_scale_dim = deq_scale_tensor->GetOriginShape().GetDims();
                if (deq_scale_dim.size() < 0 || deq_scale_dim[0] <= 1) {
                    OP_LOGD("Deq scale is a scalar, do not need to set op slice info");
                    (*inputBaseIdx) += 1;
                    run_input_update_flag = false;
                }
            } else if (!strcmp(tmpOpType.c_str(), "Elemwise")) {
                if (fusion_node->GetInDataNodes().size() <= 1) {
                    OP_LOGD("elemwise input size less than 2.");
                    run_input_update_flag = false;
                }
            } else if (!strcmp(tmpOpType.c_str(), "AscendRequantS16")) {
                is_head_fusion_ = IsHeadFusion(fusion_node, fusion_nodes);
            }
            if (run_input_update_flag) {
                (this->*opInputUpdateFunMap[tmpOpType])(split_maps, inputBaseIdx);
            }
        } else {
            OP_LOGD("op type input update fun not support by now");
        }

        // op output slice info update by op type
        if (!strcmp(tmpOpType.c_str(), "Elemwise") || !strcmp(tmpOpType.c_str(), "AscendRequantS16")) {
            if (fusion_node->GetOutDataNodes().size() <= 1) {
                OP_LOGD("out size is 1, no need to change out slice info.");
                continue;
            }
        }
        auto iterOutputUpdateFun = opOutputUpdateFunMap.find(tmpOpType);
        if (iterOutputUpdateFun != opOutputUpdateFunMap.end()) {
            (this->*opOutputUpdateFunMap[tmpOpType])(split_maps, outputBaseIdx);
        } else {
            OP_LOGD("op type output update fun not support by now");
        }
    }
    OP_LOGD("end update Conv2d fusion node slice info");
    return;
}

/******************************* del functions ******************************/

// bnreduce node need to del fmap input batch/h/w axis
void ConvSliceInfoCalBase::BNReduceSplitInfoDel(vector<AxisSplitMap> &split_maps)
{
    OP_LOGD("begin bn reduce split info del");
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_FIRST_INPUT_IDX, INPUT_BATCH_AXIS);
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_FIRST_INPUT_IDX, INPUT_H_AXIS);
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_FIRST_INPUT_IDX, INPUT_W_AXIS);
    OP_LOGD("end bn reduce split info del");
    return;
}

// quant/requant/requants16 node need to del weight input cout axis
void ConvSliceInfoCalBase::QuantAndRequantSplitInfoDel(vector<AxisSplitMap> &split_maps)
{
    OP_LOGD("begin quant and requant split info del");
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_SECOND_INPUT_IDX, OUTPUT_COUT_AXIS);
    OP_LOGD("end quant and requant split info del");
    return;
}

// stridewrite node need to del weight input cout axis and fmap input h/w axis
void ConvSliceInfoCalBase::StrideWriteSplitInfoDel(vector<AxisSplitMap> &split_maps)
{
    OP_LOGD("begin stride write split info del");
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_SECOND_INPUT_IDX, OUTPUT_COUT_AXIS);
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_FIRST_INPUT_IDX, INPUT_H_AXIS);
    ConvDelSplitInfoByInputIdxAndAxis(split_maps, CONV_FIRST_INPUT_IDX, INPUT_W_AXIS);
    OP_LOGD("end stride write split info del");
    return;
}

/******************************* output update functions ******************************/

// bnreduce node need to add other 2 or 3 outputs
void ConvSliceInfoCalBase::BnreduceOutputUpdate(vector<AxisSplitMap> &split_maps, int *baseIdxs)
{
    OP_LOGD("begin update bnreduce output slice info");
    int bnOutputSize = *baseIdxs + 1;
    for (auto &axis_split_map : split_maps) {
        auto exist_out = axis_split_map.GetOutputSplitInfoVec();
        if (!exist_out.empty() && exist_out[0].GetAxis().size() && exist_out[0].GetAxis()[0] == OUTPUT_COUT_AXIS) {
            *baseIdxs = 0;
            for(int bnOutputIdx = 1; bnOutputIdx < bnOutputSize; bnOutputIdx++) {
                (*baseIdxs) += 1;
                OutputSplitInfo output_split_info;
                if (!output_split_info.Initialize()) {
                    OP_LOGD("init output_info failed");
                    return;
                }
                output_split_info.SetIndex(*baseIdxs);
                vector<int64_t> output_vec = {1};
                output_split_info.SetAxis(output_vec);
                axis_split_map.AddOutputSplitInfo(output_split_info);
            }
        }
    }
    OP_LOGD("end update bnreduce output slice info");
    return;
}

// util output function: can add another output slice info
void ConvSliceInfoCalBase::UtilOutputUpdate(vector<AxisSplitMap> &split_maps, int *baseIdxs)
{
    OP_LOGD("begin update util output slice info");
    (*baseIdxs) += 1;
    for (auto &axis_split_map : split_maps) {
        auto exist_out = axis_split_map.GetOutputSplitInfoVec();
        if (!exist_out.empty()) {
            OutputSplitInfo output_split_info;
            if (!output_split_info.Initialize()) {
                OP_LOGD("init output_info failed");
                return;
            }
            output_split_info.SetIndex(*baseIdxs);
            auto out_axis = exist_out[0].GetAxis();
            output_split_info.SetAxis(out_axis);
            axis_split_map.AddOutputSplitInfo(output_split_info);
        }
    }
    OP_LOGD("end update util output slice info");
    return;
}

/******************************* input update functions ******************************/

// dequant input update functions: add another input info which has cout split
void ConvSliceInfoCalBase::DequantInputUpdate(vector<AxisSplitMap> &split_maps, int *inputIdxBase)
{
    OP_LOGD("begin update dequant input slice info");
    // set dequant(double input) input split info
    InputSplitInfo dequant_input_split_info;
    if (!dequant_input_split_info.Initialize()) {
        OP_LOGD("init input_info failed");
        return;
    }
    (*inputIdxBase) += 1;
    dequant_input_split_info.SetIndex(*inputIdxBase);
    vector<int64_t> axis = {OUTPUT_COUT_AXIS};
    dequant_input_split_info.SetAxis(axis);
    vector<int64_t> over_lap = {-1};
    dequant_input_split_info.SetHeadOverLap(over_lap);
    dequant_input_split_info.SetTailOverLap(over_lap);
    for (auto &axis_split_map : split_maps) {
        vector<InputSplitInfo> input_split_infos = axis_split_map.GetInputSplitInfoVec();
        for (auto input_split_info : input_split_infos) {
            vector<int64_t> axes = input_split_info.GetAxis();
            if (axes == axis) {
                axis_split_map.AddInputSplitInfo(dequant_input_split_info);
            }
        }
    }
    OP_LOGD("end update dequant input slice info");
    return;
}

// dequantS16 input update functions: add another two input info which has cout split
void ConvSliceInfoCalBase::DequantS16InputUpdate(vector<AxisSplitMap> &split_maps, int *inputIdxBase)
{
    OP_LOGD("begin update dequants16 input slice info");
    // set dequant(double input) input split info
    InputSplitInfo dequants16_input_split_info;
    if (!dequants16_input_split_info.Initialize()) {
        OP_LOGD("init input_info failed");
        return;
    }

    InputSplitInfo dequant_x1_input_split_info;
    if (!dequant_x1_input_split_info.Initialize()) {
        OP_LOGD("init input_info failed");
        return;
    }

    vector<int64_t> over_lap = {-1};
    vector<int64_t> axis = {OUTPUT_COUT_AXIS};
    (*inputIdxBase) += 1;
    dequants16_input_split_info.SetIndex(*inputIdxBase);
    dequants16_input_split_info.SetAxis(axis);
    dequants16_input_split_info.SetHeadOverLap(over_lap);
    dequants16_input_split_info.SetTailOverLap(over_lap);

    (*inputIdxBase) += 1;
    dequant_x1_input_split_info.SetIndex(*inputIdxBase);
    dequant_x1_input_split_info.SetAxis(axis);
    dequant_x1_input_split_info.SetHeadOverLap(over_lap);
    dequant_x1_input_split_info.SetTailOverLap(over_lap);
    for (auto &axis_split_map : split_maps) {
        vector<InputSplitInfo> input_split_infos = axis_split_map.GetInputSplitInfoVec();
        for (auto input_split_info : input_split_infos) {
            vector<int64_t> axes = input_split_info.GetAxis();
            if (axes == axis) {
                axis_split_map.AddInputSplitInfo(dequants16_input_split_info);
                axis_split_map.AddInputSplitInfo(dequant_x1_input_split_info);
            }
        }
    }
    OP_LOGD("end update dequants16 input slice info");
    return;
}

// requantS16 input update functions: add another two input info which one has cout split and one has all split
void ConvSliceInfoCalBase::RequantS16InputUpdate(vector<AxisSplitMap> &split_maps, int *inputIdxBase)
{
    size_t reqIndex;
    size_t x1Index;
    if (is_head_fusion_) {
        OP_LOGD("fusion mode is head fusion.");
        (*inputIdxBase) += 1;
        reqIndex = *inputIdxBase;
        (*inputIdxBase) += 1;
        x1Index = *inputIdxBase;
    } else {
        OP_LOGD("fusion mode is tail fusion.");
        (*inputIdxBase) += 1;
        x1Index = *inputIdxBase;
        (*inputIdxBase) += 1;
        reqIndex = *inputIdxBase;
    }

    InputSplitInfo requant_input_split_info;
    if (!requant_input_split_info.Initialize()) {
        OP_LOGD("init input_info failed");
        return;
    }
    requant_input_split_info.SetIndex(reqIndex);
    vector<int64_t> axis = {OUTPUT_COUT_AXIS};
    requant_input_split_info.SetAxis(axis);
    vector<int64_t> over_lap = {-1};
    requant_input_split_info.SetHeadOverLap(over_lap);
    requant_input_split_info.SetTailOverLap(over_lap);
    for (auto &axis_split_map : split_maps) {
        auto input_split_infos = axis_split_map.GetInputSplitInfoVec();
        if (!input_split_infos.empty()) {
            vector<int64_t> axes = input_split_infos[0].GetAxis();
            if (axes == axis) {
                axis_split_map.AddInputSplitInfo(requant_input_split_info);
            }
            InputSplitInfo requant_x1_input_split_info;
            if (!requant_x1_input_split_info.Initialize()) {
                OP_LOGD("init input_info failed");
                continue;
            }
            requant_x1_input_split_info.SetIndex(x1Index);
            auto x1_axis = input_split_infos[0].GetAxis();
            requant_x1_input_split_info.SetAxis(x1_axis);
            auto head_overlap = input_split_infos[0].GetHeadOverLap();
            auto tail_overlap = input_split_infos[0].GetTailOverLap();
            requant_x1_input_split_info.SetHeadOverLap(head_overlap);
            requant_x1_input_split_info.SetTailOverLap(tail_overlap);
            axis_split_map.AddInputSplitInfo(requant_x1_input_split_info);
        }
    }
    OP_LOGD("end update requants16 input slice info");
    return;
}

// elemwise input update functions: add another input info which has all split
void ConvSliceInfoCalBase::ElemwiseInputUpdate(vector<AxisSplitMap> &split_maps, int *inputIdxBase)
{
    OP_LOGD("begin update elemwise input slice info");
    (*inputIdxBase) += 1;
    for (auto &axis_split_map : split_maps) {
        auto input_split_infos = axis_split_map.GetInputSplitInfoVec();
        if (!input_split_infos.empty()) {
            InputSplitInfo elemwise_input_split_info;
            if (!elemwise_input_split_info.Initialize()) {
                OP_LOGD("init input_info failed");
            } else {
                elemwise_input_split_info.SetIndex(*inputIdxBase);
                auto in_axis = input_split_infos[0].GetAxis();
                elemwise_input_split_info.SetAxis(in_axis);
                auto head_overlap = input_split_infos[0].GetHeadOverLap();
                auto tail_overlap = input_split_infos[0].GetTailOverLap();
                elemwise_input_split_info.SetHeadOverLap(head_overlap);
                elemwise_input_split_info.SetTailOverLap(tail_overlap);
                axis_split_map.AddInputSplitInfo(elemwise_input_split_info);
            }
        }
    }
    OP_LOGD("end update elemwise input slice info");
    return;
}

// util functions
void ConvSliceInfoCalBase::ConvDelSplitInfoByInputIdxAndAxis(vector<AxisSplitMap>& split_maps, uint idx, int axis)
{
    OP_LOGD(slice_info_cal_type_.c_str(), "begin del slice info by inputIdx:%u and axis:%d", idx, axis);
    vector<AxisSplitMap> temp_maps;
    for (auto &axis_split_map : split_maps) {
        bool del_axis = false;
        auto input_split_infos = axis_split_map.GetInputSplitInfoVec();
        for (auto input_split_info : input_split_infos) {
            if (!input_split_info.GetAxis().empty()) {
                if (input_split_info.GetAxis()[0] == axis && input_split_info.GetIndex() == idx) {
                    del_axis = true;
                }
            }
        }
        if (!del_axis) {
            temp_maps.push_back(axis_split_map);
        }
    }
    split_maps = temp_maps;
    OP_LOGD("end del slice info by inputIdx and axis");
    return;
}

bool ConvSliceInfoCalBase::IsHeadFusion(const ge::NodePtr &fusion_node, const vector<ge::NodePtr> &fusion_nodes)
{
    ge::NodePtr pre_node = fusion_node->GetInDataNodes().at(0);
    if (pre_node == nullptr) {
        OP_LOGW(slice_info_cal_type_.c_str(), "first input node is nullptr");
        return false;
    }
    if (find(fusion_nodes.begin(), fusion_nodes.end(), pre_node) != fusion_nodes.end()) {
        OP_LOGD(slice_info_cal_type_.c_str(), "fusion mode is head fusion.");
        return true;
    }
    return false;
}

bool ConvSliceInfoCalBase::GetOpSliceInfoL1Fusion(OpL1FusionType& fusion_enable,
    ge::NodePtr& cube_node, const string& fusedOpType)
{
    string op_slice_info_str = "";
    if (cube_node->GetOpDesc() == nullptr) {
        OP_LOGD(fusedOpType.c_str(), "get desc failed");
        return false;
    }
    ge::AttrUtils::GetStr(cube_node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str);
    OP_LOGD(fusedOpType.c_str(), "ori _op_slice_info is %s", op_slice_info_str.c_str());
    if (op_slice_info_str.empty()) {
        OP_LOGD(fusedOpType.c_str(), "op_slice_info is null");
        return false;
    }

    OpCalcInfo op_calc_info;
    GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
    fusion_enable = op_calc_info.GetL1FusionEnable();
    return true;
}
} // namespace fe