/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

namespace {
  constexpr int64_t DENSE_DEFAULTS_INPUT_PORT_NAME_LEN = 14;
  constexpr int64_t TDENSE_INPUT_ATTR_NAME_LEN = 6;
  constexpr int64_t SPARSE_INDICES_INPUT_PORT_NAME_LEN = 14;
  constexpr int64_t NUM_SPARSE_INPUT_ATTR_NAME_LEN = 10;
  constexpr int64_t SPARSE_VALUES_INPUT_PORT_NAME_LEN = 13;
  constexpr int64_t SPARSE_TYPES_INPUT_ATTR_NAME_LEN = 12;
  constexpr int64_t SPARSE_SHAPES_INPUT_PORT_NAME_LEN = 13;
  constexpr int64_t DENSE_VALUES_INPUT_PORT_NAME_LEN = 12;
  constexpr int64_t DECODED_INDICES_INPUT_PORT_NAME_LEN = 15;
  constexpr int64_t DECODED_VALUES_INPUT_PORT_NAME_LEN = 14;
  constexpr int64_t DECODED_SHAPE_INPUT_PORT_NAME_LEN = 13;
  constexpr int64_t TOP_PATHS_INPUT_ATTR_NAME_LEN = 9;
}  // namespace

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

Status ParseSingleExampleMapping(const ge::Operator& op_src, ge::Operator& op) {
  std::vector<DynamicInputOutputInfo> value;
  DynamicInputOutputInfo input(kInput, "dense_defaults", DENSE_DEFAULTS_INPUT_PORT_NAME_LEN, "Tdense",
                               TDENSE_INPUT_ATTR_NAME_LEN);
  value.push_back(input);
  DynamicInputOutputInfo output(kOutput, "sparse_indices", SPARSE_INDICES_INPUT_PORT_NAME_LEN, "num_sparse",
                                NUM_SPARSE_INPUT_ATTR_NAME_LEN);
  value.push_back(output);
  DynamicInputOutputInfo output1(kOutput, "sparse_values", SPARSE_VALUES_INPUT_PORT_NAME_LEN, "sparse_types",
                                 SPARSE_TYPES_INPUT_ATTR_NAME_LEN);
  value.push_back(output1);
  DynamicInputOutputInfo output2(kOutput, "sparse_shapes", SPARSE_SHAPES_INPUT_PORT_NAME_LEN, "num_sparse",
                                 NUM_SPARSE_INPUT_ATTR_NAME_LEN);
  value.push_back(output2);
  DynamicInputOutputInfo output3(kOutput, "dense_values", DENSE_VALUES_INPUT_PORT_NAME_LEN, "Tdense",
                                 TDENSE_INPUT_ATTR_NAME_LEN);
  value.push_back(output3);
  AutoMappingByOpFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register ParseSingleExample op to GE
REGISTER_CUSTOM_OP("ParseSingleExample")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParseSingleExample")
    .ParseParamsByOperatorFn(ParseSingleExampleMapping)
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
  // Add dynamic input/output to ir, if 4th and 5th param is -1 means add input/output from back
  // if 1 means add input/output from head
  AutoMappingFnDynamic(op_src, op, value, -1, 1);
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

Status MappingFnParseExample(const google::protobuf::Message* op_src,
                             ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("sparse_keys", "Nsparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["in"] = pair<string, string>("dense_keys", "Ndense");
  AutoMappingFnDynamic(op_src, op, value);
  value["in"] = pair<string, string>("dense_defaults", "Ndense");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("sparse_indices", "Nsparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("sparse_values", "Nsparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("sparse_shapes", "Nsparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("dense_values", "Ndense");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register ParseExample op to GE
REGISTER_CUSTOM_OP("ParseExample")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParseExample")
    .ParseParamsFn(MappingFnParseExample)
    .ImplyType(ImplyType::AI_CPU);

Status MappingFnParseSingleSequenceExample(
    const google::protobuf::Message* op_src, ge::Operator& op) {
  map<string, pair<string, string>> value;
  value["in"] = pair<string, string>("context_sparse_keys", "Ncontext_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["in"] = pair<string, string>("context_dense_keys", "Ncontext_dense");
  AutoMappingFnDynamic(op_src, op, value);
  value["in"] =
      pair<string, string>("feature_list_sparse_keys", "Nfeature_list_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["in"] =
      pair<string, string>("feature_list_dense_keys", "Nfeature_list_dense");
  AutoMappingFnDynamic(op_src, op, value);
  value["in"] =
      pair<string, string>("context_dense_defaults", "Ncontext_dense");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] =
      pair<string, string>("context_sparse_indices", "Ncontext_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] =
      pair<string, string>("context_sparse_values", "Ncontext_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] =
      pair<string, string>("context_sparse_shapes", "Ncontext_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("context_dense_values", "Ncontext_dense");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("feature_list_sparse_indices",
                                      "Nfeature_list_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("feature_list_sparse_values",
                                      "Nfeature_list_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] = pair<string, string>("feature_list_sparse_shapes",
                                      "Nfeature_list_sparse");
  AutoMappingFnDynamic(op_src, op, value);
  value["out"] =
      pair<string, string>("feature_list_dense_values", "Nfeature_list_dense");
  AutoMappingFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register ParseExample op to GE
REGISTER_CUSTOM_OP("ParseSingleSequenceExample")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParseSingleSequenceExample")
    .ParseParamsFn(MappingFnParseSingleSequenceExample)
    .ImplyType(ImplyType::AI_CPU);

Status MappingFnCTCBeamSearchDecoder(const ge::Operator& op_src, ge::Operator& op) {
  std::vector<DynamicInputOutputInfo> value;
  DynamicInputOutputInfo output(kOutput, "decoded_indices", DECODED_INDICES_INPUT_PORT_NAME_LEN, "top_paths",
                                TOP_PATHS_INPUT_ATTR_NAME_LEN);
  value.push_back(output);
  DynamicInputOutputInfo output1(kOutput, "decoded_values", DECODED_VALUES_INPUT_PORT_NAME_LEN, "top_paths",
                                 TOP_PATHS_INPUT_ATTR_NAME_LEN);
  value.push_back(output1);
  DynamicInputOutputInfo output2(kOutput, "decoded_shape", DECODED_SHAPE_INPUT_PORT_NAME_LEN, "top_paths",
                                 TOP_PATHS_INPUT_ATTR_NAME_LEN);
  value.push_back(output2);
  AutoMappingByOpFnDynamic(op_src, op, value);
  return SUCCESS;
}

// register CTCBeamSearchDecoder op to GE
REGISTER_CUSTOM_OP("CTCBeamSearchDecoder")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("CTCBeamSearchDecoder")
    .ParseParamsByOperatorFn(MappingFnCTCBeamSearchDecoder)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
