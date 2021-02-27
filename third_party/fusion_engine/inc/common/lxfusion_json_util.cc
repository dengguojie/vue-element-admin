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

#include "lxfusion_json_util.h"
#include <nlohmann/json.hpp>
#include "graph/debug/ge_attr_define.h"

namespace fe {
    const std::string InputSplitInfo_Idx = "idx";
    const std::string InputSplitInfo_Axis = "axis";
    const std::string InputSplitInfo_HeadOverLap = "headOverLap";
    const std::string InputSplitInfo_TailOverLap = "tailOverLap";

    const std::string OutputSplitInfo_Idx = "idx";
    const std::string OutputSplitInfo_Axis = "axis";

    const std::string AxisSplitMap_InputList = "inputList";
    const std::string AxisSplitMap_OutputList = "outputList";

    const std::string InputReduceInfo_Idx = "idx";
    const std::string InputReduceInfo_Axis = "axis";

    const std::string OutputReduceInfo_Idx = "idx";
    const std::string OutputReduceInfo_ReduceType = "reduceType";
    const std::string OutputReduceInfo_IsAtomic = "isAtomic";

    const std::string AxisReduceMap_InputList = "inputList";
    const std::string AxisReduceMap_OutputList = "outputList";

    const std::string OpCalcInfo_SplitMaps = "splitMaps";
    const std::string OpCalcInfo_ReduceMaps = "reduceMaps";
    const std::string OpCalcInfo_L1FusionEnable = "l1FusionEnable";
    const std::string OpCalcInfo_MinTbeL1Space = "minTbeL1Space";

    using OpCalcInfoPtr = std::shared_ptr<OpCalcInfo>;

    void from_json(const nlohmann::json& json_value, InputSplitInfo& input_split_info) {
        auto idx = json_value.at(InputSplitInfo_Idx).get<size_t>();
        input_split_info.SetIndex(idx);
        auto axis = json_value.at(InputSplitInfo_Axis).get<std::vector<int64_t>>();
        input_split_info.SetAxis(axis);
        auto head_over_lap = json_value.at(InputSplitInfo_HeadOverLap).get<std::vector<int64_t>>();
        input_split_info.SetHeadOverLap(head_over_lap);
        auto tail_over_lap = json_value.at(InputSplitInfo_TailOverLap).get<std::vector<int64_t>>();
        input_split_info.SetTailOverLap(tail_over_lap);
    }

    void from_json(const nlohmann::json& json_value, OutputSplitInfo& output_split_info) {
        auto idx = json_value.at(OutputReduceInfo_Idx).get<size_t>();
        output_split_info.SetIndex(idx);
        auto axis = json_value.at(OutputSplitInfo_Axis).get<std::vector<int64_t>>();
        output_split_info.SetAxis(axis);
    }

    void from_json(const nlohmann::json& json_value, AxisSplitMap& axis_split_map) {
        auto input_list = json_value.at(AxisSplitMap_InputList).get<std::vector<InputSplitInfo>>();
        axis_split_map.SetInputSplitInfos(input_list);
        auto output_list = json_value.at(AxisSplitMap_OutputList).get<std::vector<OutputSplitInfo>>();
        axis_split_map.SetOutputSplitInfos(output_list);
    }

    void from_json(const nlohmann::json& json_value, InputReduceInfo& input_reduce_info) {
        auto idx = json_value.at(InputReduceInfo_Idx).get<size_t>();
        input_reduce_info.SetIndex(idx);
        auto axis = json_value.at(InputReduceInfo_Axis).get<std::vector<int64_t>>();
        input_reduce_info.SetAxis(axis);
    }

    void from_json(const nlohmann::json& json_value, OutputReduceInfo& output_reduce_info) {
        auto idx = json_value.at(OutputReduceInfo_Idx).get<size_t>();
        output_reduce_info.SetIndex(idx);
        auto reduce_type = json_value.at(OutputReduceInfo_ReduceType).get<OpReduceType>();
        output_reduce_info.SetReduceType(reduce_type);
        auto is_atomic = json_value.at(OutputReduceInfo_IsAtomic).get<bool>();
        output_reduce_info.SetIsAtomic(is_atomic);
    }

    void from_json(const nlohmann::json& json_value, AxisReduceMap& axis_reduce_map) {
        auto input_list = json_value.at(AxisReduceMap_InputList).get<std::vector<InputReduceInfo>>();
        axis_reduce_map.SetInputReduceInfos(input_list);
        auto output_list = json_value.at(AxisReduceMap_OutputList).get<std::vector<OutputReduceInfo>>();
        axis_reduce_map.SetOutputReduceInfos(output_list);
    }

    void from_json(const nlohmann::json& json_value, OpCalcInfo& op_calc_info) {
        auto split_maps = json_value.at(OpCalcInfo_SplitMaps).get<std::vector<AxisSplitMap>>();
        op_calc_info.SetAxisSplitMaps(split_maps);
        auto reduce_maps = json_value.at(OpCalcInfo_ReduceMaps).get<std::vector<AxisReduceMap>>();
        op_calc_info.SetAxisReduceMaps(reduce_maps);
        auto l1_fusion_enable = json_value.at(OpCalcInfo_L1FusionEnable).get<OpL1FusionType>();
        op_calc_info.SetL1FusionEnable(l1_fusion_enable);
        auto min_tbe_l1_space = json_value.at(OpCalcInfo_MinTbeL1Space).get<int64_t>();
        op_calc_info.SetMinTbeL1Space(min_tbe_l1_space);
    }

    void to_json(nlohmann::json& json_value, const InputSplitInfo& input_split_info) {
        json_value = nlohmann::json{{InputSplitInfo_Idx, input_split_info.GetIndex()},
                                    {InputSplitInfo_Axis, input_split_info.GetAxis()},
                                    {InputSplitInfo_HeadOverLap, input_split_info.GetHeadOverLap()},
                                    {InputSplitInfo_TailOverLap, input_split_info.GetTailOverLap()}};
    }

    void to_json(nlohmann::json& json_value, const OutputSplitInfo& output_split_info){
        json_value = nlohmann::json{{OutputSplitInfo_Idx, output_split_info.GetIndex()},
                                    {OutputSplitInfo_Axis, output_split_info.GetAxis()}};
    }

    void to_json(nlohmann::json& json_value, const AxisSplitMap& axis_split_map) {
        json_value = nlohmann::json{{AxisSplitMap_InputList, axis_split_map.GetInputSplitInfoVec()},
                                    {AxisSplitMap_OutputList, axis_split_map.GetOutputSplitInfoVec()}};
    }

    void to_json(nlohmann::json& json_value, const InputReduceInfo& input_reduce_info) {
        json_value = nlohmann::json{{InputReduceInfo_Idx, input_reduce_info.GetIndex()},
                                    {InputReduceInfo_Axis, input_reduce_info.GetAxis()}};
    }

    void to_json(nlohmann::json& json_value, const OutputReduceInfo& output_reduce_info) {
        json_value = nlohmann::json{{OutputReduceInfo_Idx, output_reduce_info.GetIndex()},
                                    {OutputReduceInfo_ReduceType, output_reduce_info.GetReduceType()},
                                    {OutputReduceInfo_IsAtomic, output_reduce_info.GetIsAtomic()}};
    }

    void to_json(nlohmann::json& json_value, const AxisReduceMap& axis_reduce_map) {
        json_value = nlohmann::json{{AxisReduceMap_InputList, axis_reduce_map.GetInputReduceInfoVec()},
                                    {AxisReduceMap_OutputList, axis_reduce_map.GetOutputReduceInfoVec()}};
    }

    void to_json(nlohmann::json& json_value, const OpCalcInfo& op_calc_info) {
        json_value = nlohmann::json{{OpCalcInfo_SplitMaps, op_calc_info.GetAxisSplitMapVec()},
                                    {OpCalcInfo_ReduceMaps, op_calc_info.GetAxisReduceMapVec()},
                                    {OpCalcInfo_L1FusionEnable, op_calc_info.GetL1FusionEnable()},
                                    {OpCalcInfo_MinTbeL1Space, op_calc_info.GetMinTbeL1Space()}};
    }

    void SetOpSliceInfoToJson(fe::OpCalcInfo& op_calc_info, std::string & op_calc_info_str) {
        nlohmann::json l1_info_json = nlohmann::json{{fe::OP_SLICE_INFO, op_calc_info}};
        op_calc_info_str = l1_info_json.dump();
        CM_LOGD("set op_slice_info is %s", op_calc_info_str.c_str());
    }

    void GetOpSliceInfoFromJson(fe::OpCalcInfo& op_calc_info, std::string & op_calc_info_str) {
        nlohmann::json op_calc_info_json = nlohmann::json::parse(op_calc_info_str);
        op_calc_info_json.at(fe::OP_SLICE_INFO).get_to(op_calc_info);
        CM_LOGD("get op_slice_info is %s", op_calc_info_str.c_str());
    }

    void SetFusionOpSliceInfoToJson(fe::OpCalcInfo& op_calc_info, std::string & op_calc_info_str) {
        nlohmann::json l1_info_json = nlohmann::json{{fe::FUSION_OP_SLICE_INFO, op_calc_info}};
        op_calc_info_str = l1_info_json.dump();
        CM_LOGD("set op_slice_info is %s", op_calc_info_str.c_str());
    }
}   