/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file dynamic_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// register QueueDequeueUpTo op to GE
Status QueueDequeueUpToMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("components", "component_types");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("QueueDequeueUpTo")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("QueueDequeueUpToV2")
    .ParseParamsFn(QueueDequeueUpToMapping)
    .ImplyType(ImplyType::AI_CPU);

// register Stage op to GE
Status StageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Stage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Stage")
    .ParseParamsFn(StageMapping)
    .ImplyType(ImplyType::AI_CPU);

// register StagePeek op to GE
Status StagePeekMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("y", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("StagePeek")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StagePeek")
    .ParseParamsFn(StagePeekMapping)
    .ImplyType(ImplyType::AI_CPU);

// register StringFormat op to GE
Status StringFormatMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("x", "T");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("StringFormat")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringFormat")
    .ParseParamsFn(StringFormatMapping)
    .ImplyType(ImplyType::AI_CPU);

// register StringJoin op to GE
Status StringJoinMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("x", "N");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("StringJoin")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("StringJoin")
    .ParseParamsFn(StringJoinMapping)
    .ImplyType(ImplyType::AI_CPU);

// register Unstage op to GE
Status UnstageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("y", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Unstage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Unstage")
    .ParseParamsFn(UnstageMapping)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapUnstageNoKey op to GE
Status OrderedMapUnstageNoKeyMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("OrderedMapUnstageNoKey")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapUnstageNoKey")
    .ParseParamsFn(OrderedMapUnstageNoKeyMapping)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapUnstage op to GE
Status OrderedMapUnstageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("OrderedMapUnstage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapUnstage")
    .ParseParamsFn(OrderedMapUnstageMapping)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapStage op to GE
Status OrderedMapStageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("values", "fake_dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("OrderedMapStage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapStage")
    .ParseParamsFn(OrderedMapStageMapping)
    .ImplyType(ImplyType::AI_CPU);

// register OrderedMapPeek op to GE
Status OrderedMapPeekMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("OrderedMapPeek")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("OrderedMapPeek")
    .ParseParamsFn(OrderedMapPeekMapping)
    .ImplyType(ImplyType::AI_CPU);

// register MapUnstageNoKey op to GE
Status MapUnstageNoKeyMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MapUnstageNoKey")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapUnstageNoKey")
    .ParseParamsFn(MapUnstageNoKeyMapping)
    .ImplyType(ImplyType::AI_CPU);

// register MapUnstage op to GE
Status MapUnstageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MapUnstage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapUnstage")
    .ParseParamsFn(MapUnstageMapping)
    .ImplyType(ImplyType::AI_CPU);

// register MapStage op to GE
Status MapStageMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("values", "fake_dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MapStage")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapStage")
    .ParseParamsFn(MapStageMapping)
    .ImplyType(ImplyType::AI_CPU);

// register MapPeek op to GE
Status MapPeekMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "dtypes");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("MapPeek")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MapPeek")
    .ParseParamsFn(MapPeekMapping)
    .ImplyType(ImplyType::AI_CPU);

// register DynamicPartition op to GE
Status DynamicPartitionMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("y", "num_partitions");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicPartition")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicPartition")
    .ParseParamsFn(DynamicPartitionMapping)
    .ImplyType(ImplyType::AI_CPU);

// register Batch op to GE
Status BatchMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("x_tensors", "T");
  value["out"] = pair<string, string>("y_tensors", "T");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Batch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Batch")
    .ParseParamsFn(BatchMapping)
    .ImplyType(ImplyType::AI_CPU);

// register BarrierTakeMany op to GE
Status BarrierTakeManyMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["out"] = pair<string, string>("values", "component_types");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BarrierTakeMany")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BarrierTakeMany")
    .ParseParamsFn(BarrierTakeManyMapping)
    .ImplyType(ImplyType::AI_CPU);

// register RaggedGather op to GE
Status RaggedGatherMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("params_nested_splits", "PARAMS_RAGGED_RANK");
  value["out"] = pair<string, string>("output_nested_splits", "OUTPUT_RAGGED_RANK");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("RaggedGather")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedGather")
    .ParseParamsFn(RaggedGatherMapping)
    .ImplyType(ImplyType::AI_CPU);

// register RaggedTensorToSparse op to GE
Status RaggedTensorToSparseMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("rt_nested_splits", "RAGGED_RANK");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("RaggedTensorToSparse")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedTensorToSparse")
    .ParseParamsFn(RaggedTensorToSparseMapping)
    .ImplyType(ImplyType::AI_CPU);

// register RaggedTensorToTensor op to GE
Status RaggedTensorToTensorMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("row_partition_tensors", "num_row_partition_tensors");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("RaggedTensorToTensor")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedTensorToTensor")
    .ParseParamsFn(RaggedTensorToTensorMapping)
    .ImplyType(ImplyType::AI_CPU);

}  // namespace domi
