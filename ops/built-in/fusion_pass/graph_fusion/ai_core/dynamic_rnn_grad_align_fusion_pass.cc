/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_rnn_grad_align_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "DynamicRNNGrad";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNGrad";

static map<std::string, int> INPUT_INDEX = {
    {"x", 0},  {"w", 1},  {"b", 2},   {"y", 3},  {"init_h", 4}, {"init_c", 5}, {"h", 6},  {"c", 7},
    {"dy", 8}, {"dh", 9}, {"dc", 10}, {"i", 11}, {"j", 12},     {"f", 13},     {"o", 14}, {"tanhct", 15}};

static map<std::string, int> OUTPUT_INDEX = {{"dw", 0}, {"db", 1}, {"dx", 2}, {"dh_prev", 3}, {"dc_prev", 4}};

static map<std::string, int> CELL_INPUT_INDEX = {
    {"c",      0},
    {"dy",     1},
    {"dh",     2},
    {"dc",     3},
    {"i",      4},
    {"j",      5},
    {"f",      6},
    {"o",      7},
    {"tanhct", 8}};

vector<FusionPattern*> DynamicRNNGradAlignFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNGradAlignFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::GetConstNodeOne(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::GeTensorPtr assitPtr = nullptr;
  int64_t matrixSize = t_dim * batch_nz_dim * 16 * 16;
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[matrixSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT("DynamicRnnGrad", "The inputAssit is NULL"),
                    failStatus = true);
  auto retMem = memset_s(inputAssit.get(), matrixSize, 1, matrixSize);
  FUSION_PASS_CHECK(retMem != EOK,
                    VECTOR_FUSION_INNER_ERR_REPORT("DynamicRnnGrad", "Failed to operate memset_s function."),
                    failStatus = true);
  float* dstConst = inputAssit.get();
  for (int j = 0; j < matrixSize; j++) {
    *(dstConst + j) = 1;
  }

  ge::GeTensorDesc tensorDesc;
  vector<int64_t> assit_dim_info = {};
  vector<int64_t> assit_dim_info_origin = {};
  assit_dim_info = {t_dim, batch_nz_dim, 1, 16, 16};
  assit_dim_info_origin = {t_dim, 1, batch_dim};

  ge::GeShape assit_shape(assit_dim_info);
  ge::GeShape assit_shape_origin(assit_dim_info_origin);

  tensorDesc.SetShape(assit_shape);
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  tensorDesc.SetOriginFormat(ge::FORMAT_ND);
  tensorDesc.SetOriginShape(assit_shape_origin);
  tensorDesc.SetOriginDataType(ge::DT_FLOAT);

  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), matrixSize * sizeof(float))),
                          failStatus = true;
                          return nullptr);

  ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(assitPtr);
  ge::NodePtr const_node = graph.AddNode(const_opdesc);
  newNodes.push_back(const_node);

  return const_node;
}

vector<vector<ge::NodePtr>> DynamicRNNGradAlignFusionPass::AddTLoopNode(ge::NodePtr dynamicRNNGradNode,
                                                                        ge::ComputeGraph& graph,
                                                                        vector<ge::NodePtr>& newNodes,
                                                                        bool& failStatus) {
  vector<vector<ge::NodePtr>> result = {};
  vector<ge::NodePtr> basicLstm_cell_state_grad_nodes = {};
  vector<ge::NodePtr> matmul_nodes = {};
  vector<ge::NodePtr> split_nodes = {};
  vector<ge::NodePtr> reshape_nodes = {};

  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  int64_t t_size = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dy"]).GetShape().GetDim(0);
  // all input
  ge::GeTensorDesc inputC = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["c"]);
  ge::GeTensorDesc inputDy = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dy"]);
  ge::GeTensorDesc inputDh = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dh"]);
  ge::GeTensorDesc inputI = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["i"]);
  // add split op

  for (int64_t i = 0; i < t_size; i++) {
    // add state_gate op
    OP_LOGD(FUSED_OP_TYPE.c_str(), "start add Cell node for loop:%d.", i);
    ge::OpDescPtr basicLstmCellStateGradDesc = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (basicLstmCellStateGradDesc = std::make_shared<ge::OpDesc>(
             dynamicRNNGradDesc->GetName() + "/LstmInputGrad/BasicLSTMCellCStateGrad" + std::to_string(i),
             "BasicLSTMCellCStateGradV2")),
        failStatus = true;
        return result);
    // set state_gate op input
    SetInputDescForGradCell(dynamicRNNGradDesc, inputC, inputDy, inputI, i, basicLstmCellStateGradDesc);

    // set state_gate op output
    vector<int64_t> output_dims = getOutputDimsForGradCell(basicLstmCellStateGradDesc);
    ge::GeShape output_shape(output_dims);
    GeTensorDesc output_tensor_desc = SetOutputDescForGradCell(inputI, basicLstmCellStateGradDesc, output_shape);

    // set state_gate op attr
    ge::AttrUtils::SetFloat(basicLstmCellStateGradDesc, "forget_bias", 1.0);
    ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "activation", "Tanh");

    // add reshape
    string reshapeNodeName = dynamicRNNGradNode->GetName() + "/cellReshape" + std::to_string(i);
    auto reshapeOp = ge::OperatorFactory::CreateOperator(reshapeNodeName.c_str(), "Reshape");
    FUSION_PASS_CHECK(reshapeOp.IsEmpty(),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create Reshape Op operator error."),
                      return result);
    auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
    reshapeOp.BreakConnect();

    SetReshapeDescForCell(output_dims, output_tensor_desc, reshape_desc);

    ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);
    FUSION_PASS_CHECK(myReshape_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "create Reshape node  error."),
                      return result);
    newNodes.push_back(myReshape_node);
    reshape_nodes.push_back(myReshape_node);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "start add matmul node for loop:%d.", i);
    // add matmul
    ge::OpDescPtr lstmBatchMatMulDesc = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (lstmBatchMatMulDesc = std::make_shared<ge::OpDesc>(
             dynamicRNNGradDesc->GetName() + "/LstmInputGrad/Matmul" + std::to_string(i), "BatchMatMulV2")),
        failStatus = true;
        return result);
    vector<int64_t> outputy_dims;
    AddBatchMatMulForCell(output_shape, lstmBatchMatMulDesc, outputy_dims);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "start add splitVD node for loop:%d.", i);
    // add split op
    ge::OpDescPtr lstmSplitDesc = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (lstmSplitDesc = std::make_shared<ge::OpDesc>(
             dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVD" + std::to_string(i), "SplitVD")),
        failStatus = true;
        return result);
    lstmSplitDesc = AddSpiltForCell(outputy_dims, lstmSplitDesc);

    ge::NodePtr basicLstmCellStateGradNode = graph.AddNode(basicLstmCellStateGradDesc);
    FUSION_PASS_CHECK(basicLstmCellStateGradNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:basicLstmCellStateGradNode is null, fusion failed."),
                      failStatus = true);
    basicLstm_cell_state_grad_nodes.push_back(basicLstmCellStateGradNode);
    newNodes.push_back(basicLstmCellStateGradNode);

    ge::NodePtr matmulNode = graph.AddNode(lstmBatchMatMulDesc);
    FUSION_PASS_CHECK(matmulNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:matmulNode is null, fusion failed."),
                      failStatus = true);
    matmul_nodes.push_back(matmulNode);
    newNodes.push_back(matmulNode);

    ge::NodePtr splitNode = graph.AddNode(lstmSplitDesc);
    FUSION_PASS_CHECK(splitNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:splitNode is null, fusion failed."),
                      failStatus = true);
    split_nodes.push_back(splitNode);
    newNodes.push_back(splitNode);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "end add node for loop.");
  result.push_back(basicLstm_cell_state_grad_nodes);
  result.push_back(matmul_nodes);
  result.push_back(split_nodes);
  result.push_back(reshape_nodes);

  return result;
}

vector<int64_t> DynamicRNNGradAlignFusionPass::getOutputDimsForGradCell(
    const OpDescPtr& basicLstmCellStateGradDesc) const {
  GeTensorDesc input_tensor_desc_c = basicLstmCellStateGradDesc->GetInputDesc(3);
  vector<int64_t> output_dims;
  output_dims.push_back(input_tensor_desc_c.GetShape().GetDim(0));
  output_dims.push_back(4 * ((((input_tensor_desc_c.GetShape().GetDim(1)) + 15) / 16) * 16));
  return output_dims;
}

GeTensorDesc DynamicRNNGradAlignFusionPass::SetOutputDescForGradCell(const GeTensorDesc& inputI,
                                                                     OpDescPtr& basicLstmCellStateGradDesc,
                                                                     const GeShape& output_shape) const {
  GeTensorDesc output_tensor_desc = GeTensorDesc(output_shape, FORMAT_ND, DT_FLOAT16);
  output_tensor_desc.SetOriginShape(output_shape);
  output_tensor_desc.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddOutputDesc("dgate", output_tensor_desc);

  vector<int64_t> dc_prev_tensor_dims;
  dc_prev_tensor_dims.push_back(inputI.GetShape().GetDim(1));
  dc_prev_tensor_dims.push_back(inputI.GetShape().GetDim(2));
  GeShape dc_prev_tensor_shape(dc_prev_tensor_dims);
  GeTensorDesc tensor_dc_prew = GeTensorDesc(dc_prev_tensor_shape, FORMAT_ND, DT_FLOAT);
  tensor_dc_prew.SetOriginShape(dc_prev_tensor_shape);
  tensor_dc_prew.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddOutputDesc("dc_prev", tensor_dc_prew);
  return output_tensor_desc;
}

void DynamicRNNGradAlignFusionPass::SetReshapeDescForCell(const vector<int64_t>& output_dims,
                                                          const GeTensorDesc& output_tensor_desc,
                                                          OpDescPtr& reshape_desc) const {
  vector<int64_t> inputTensorDescCellDgateDims = {(output_dims[1] + 15) / 16, (output_dims[0] + 15) / 16, 16, 16};
  vector<int64_t> inputTensorDescCellDgateOriDims = {output_dims[0], output_dims[1]};
  GeShape inputTensorDescCellDgateShape(inputTensorDescCellDgateDims);

  GeTensorDesc reshapeCellInputDesc =
      GeTensorDesc(inputTensorDescCellDgateShape, FORMAT_FRACTAL_NZ, output_tensor_desc.GetDataType());
  reshapeCellInputDesc.SetOriginShape(GeShape(inputTensorDescCellDgateOriDims));
  reshapeCellInputDesc.SetOriginFormat(FORMAT_ND);

  vector<int64_t> outputTensorDescCellDgateDims = {1, (output_dims[1] + 15) / 16, (output_dims[0] + 15) / 16, 16, 16};
  vector<int64_t> outputTensorDescCellDgateOriDims = {1, output_dims[0], output_dims[1]};
  GeShape outputTensorDescCellDgateShape(outputTensorDescCellDgateDims);

  GeTensorDesc reshapeCellOutputDesc =
      GeTensorDesc(outputTensorDescCellDgateShape, FORMAT_FRACTAL_NZ, output_tensor_desc.GetDataType());
  reshapeCellOutputDesc.SetOriginShape(GeShape(outputTensorDescCellDgateOriDims));
  reshapeCellOutputDesc.SetOriginFormat(FORMAT_ND);

  reshape_desc->UpdateInputDesc("x", reshapeCellInputDesc);
  reshape_desc->UpdateOutputDesc("y", reshapeCellOutputDesc);
}

void DynamicRNNGradAlignFusionPass::SetInputDescForGradCell(
    const OpDescPtr& dynamicRNNGradDesc, const GeTensorDesc& inputC, const GeTensorDesc& inputDy,
    const GeTensorDesc& inputI, int64_t i,
    OpDescPtr& basicLstmCellStateGradDesc) const {  // add state_gate op input
  vector<int64_t> cur_tensor_dims;
  cur_tensor_dims.push_back(1);
  cur_tensor_dims.push_back(inputC.GetShape().GetDim(1));
  cur_tensor_dims.push_back(inputC.GetShape().GetDim(2));
  GeShape cur_tensorc_shape(cur_tensor_dims);
  GeShape cur_tensorc_original_shape(cur_tensor_dims);
  GeTensorDesc cur_tensorc = GeTensorDesc(cur_tensorc_shape, FORMAT_ND, DT_FLOAT);
  cur_tensorc.SetOriginShape(cur_tensorc_original_shape);
  cur_tensorc.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("c", cur_tensorc);

  vector<int64_t> cur_tensor_dy_dims;
  cur_tensor_dy_dims.push_back(1);
  cur_tensor_dy_dims.push_back(inputDy.GetShape().GetDim(1));
  cur_tensor_dy_dims.push_back(inputDy.GetShape().GetDim(2));
  GeShape cur_tensor_dy_shape(cur_tensor_dy_dims);
  GeShape cur_tensor_dy_original_shape(cur_tensor_dy_dims);
  GeTensorDesc cur_tensor_dy = GeTensorDesc(cur_tensor_dy_shape, FORMAT_ND, DT_FLOAT);
  cur_tensor_dy.SetOriginShape(cur_tensor_dy_original_shape);
  cur_tensor_dy.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("dy", cur_tensor_dy);

  if (i != 0) {
    vector<int64_t> cur_tensor_dh_dims;
    cur_tensor_dh_dims.push_back(1);
    cur_tensor_dh_dims.push_back(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dh"]).GetShape().GetDim(0));
    cur_tensor_dh_dims.push_back(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dh"]).GetShape().GetDim(1));
    GeShape cur_tensor_dh_shape(cur_tensor_dh_dims);
    GeTensorDesc dh_input_tensor_desc = GeTensorDesc(cur_tensor_dh_shape, FORMAT_ND, DT_FLOAT);
    dh_input_tensor_desc.SetOriginShape(cur_tensor_dh_shape);
    dh_input_tensor_desc.SetOriginFormat(FORMAT_ND);
    basicLstmCellStateGradDesc->AddInputDesc("dh", dh_input_tensor_desc);
  } else {
    GeTensorDesc dh_input_tensor_desc =
        GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dh"]).GetShape(), FORMAT_ND, DT_FLOAT);
    dh_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dh"]).GetShape());
    dh_input_tensor_desc.SetOriginFormat(FORMAT_ND);
    basicLstmCellStateGradDesc->AddInputDesc("dh", dh_input_tensor_desc);
  }

  GeTensorDesc dc_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dc"]).GetShape(), FORMAT_ND, DT_FLOAT);
  dc_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dc"]).GetShape());
  dc_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("dc", dc_input_tensor_desc);

  vector<int64_t> cur_tensori_dims;
  cur_tensori_dims.push_back(1);
  cur_tensori_dims.push_back(inputI.GetShape().GetDim(1));
  cur_tensori_dims.push_back(inputI.GetShape().GetDim(2));
  GeShape cur_tensori_shape(cur_tensori_dims);
  GeTensorDesc cur_tensori = GeTensorDesc(cur_tensori_shape, FORMAT_ND, DT_FLOAT);
  cur_tensori.SetOriginShape(cur_tensorc_original_shape);
  cur_tensori.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("i", cur_tensori);

  GeTensorDesc cur_tensorj = GeTensorDesc(cur_tensori_shape, FORMAT_ND, DT_FLOAT);
  cur_tensorj.SetOriginShape(cur_tensorc_original_shape);
  cur_tensorj.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("j", cur_tensorj);

  GeTensorDesc cur_tensorf = GeTensorDesc(cur_tensori_shape, FORMAT_ND, DT_FLOAT);
  cur_tensorf.SetOriginShape(cur_tensorc_original_shape);
  cur_tensorf.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("f", cur_tensorf);

  GeTensorDesc cur_tensoro = GeTensorDesc(cur_tensori_shape, FORMAT_ND, DT_FLOAT);
  cur_tensoro.SetOriginShape(cur_tensorc_original_shape);
  cur_tensoro.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("o", cur_tensoro);

  GeTensorDesc cur_tensor_tanhct = GeTensorDesc(cur_tensori_shape, FORMAT_ND, DT_FLOAT);
  cur_tensor_tanhct.SetOriginShape(cur_tensorc_original_shape);
  cur_tensor_tanhct.SetOriginFormat(FORMAT_ND);
  basicLstmCellStateGradDesc->AddInputDesc("tanhct", cur_tensor_tanhct);
}

OpDescPtr& DynamicRNNGradAlignFusionPass::AddSpiltForCell(vector<int64_t>& outputy_dims,
                                                          OpDescPtr& lstmSplitDesc) const {
  GeTensorDesc split_input_tensor_desc = GeTensorDesc(GeShape(outputy_dims), FORMAT_FRACTAL_NZ, DT_FLOAT);
  split_input_tensor_desc.SetOriginShape(GeShape(outputy_dims));
  split_input_tensor_desc.SetOriginFormat(FORMAT_FRACTAL_NZ);

  lstmSplitDesc->AddInputDesc("y", split_input_tensor_desc);

  vector<int64_t> dx_dims{1, input_nz_dim, batch_nz_dim, 16, 16};
  GeShape dx_shape(dx_dims);
  vector<int64_t> dx_ori_dims{1, batch_dim, input_dim};
  GeShape dx_original_shape(dx_ori_dims);
  GeTensorDesc tensor_dx = GeTensorDesc(dx_shape, FORMAT_FRACTAL_NZ, DT_FLOAT);

  if (tSizeJudge == 1) {
    tensor_dx.SetOriginShape(dx_original_shape);
    tensor_dx.SetOriginFormat(FORMAT_ND);
  } else {
    tensor_dx.SetOriginShape(dx_shape);
    tensor_dx.SetOriginFormat(FORMAT_FRACTAL_NZ);
  }

  lstmSplitDesc->AddOutputDesc("dx", tensor_dx);

  vector<int64_t> dh_dims{1, hidden_nz_dim, batch_nz_dim, 16, 16};
  GeShape dh_shape(dh_dims);
  GeTensorDesc dh_tensor_desc = GeTensorDesc(dh_shape, FORMAT_FRACTAL_NZ, DT_FLOAT);
  vector<int64_t> dh_ori_dims{1, batch_dim, hidden_dim};
  GeShape dh_ori_shape(dh_ori_dims);
  dh_tensor_desc.SetOriginShape(dh_ori_shape);
  dh_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitDesc->AddOutputDesc("dh_prev", dh_tensor_desc);

  vector<int64_t> size_splits;
  size_splits.push_back(input_nz_dim);
  size_splits.push_back(hidden_nz_dim);
  AttrUtils::SetListInt(lstmSplitDesc, "size_splits", size_splits);
  AttrUtils::SetInt(lstmSplitDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);
  return lstmSplitDesc;
}

void DynamicRNNGradAlignFusionPass::AddBatchMatMulForCell(const GeShape& output_origin_shape,
                                                          OpDescPtr& lstmBatchMatMulDesc,
                                                          vector<int64_t>& outputy_dims) const {  // add matmul input
  vector<int64_t> LeftDims{hidden_nz_dim * 4, batch_nz_dim, 16, 16};
  GeTensorDesc left_tensor_desc = GeTensorDesc(GeShape(LeftDims), FORMAT_FRACTAL_NZ, DT_FLOAT16);
  left_tensor_desc.SetOriginShape(output_origin_shape);
  left_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmBatchMatMulDesc->AddInputDesc("dgate", left_tensor_desc);

  vector<int64_t> WeightDims{input_nz_dim + hidden_nz_dim, hidden_nz_dim * 4, 16, 16};
  vector<int64_t> WeightoriDims{input_dim + hidden_dim, 4 * hidden_dim};
  GeTensorDesc weight_tensor_desc = GeTensorDesc(GeShape(WeightDims), FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16);
  weight_tensor_desc.SetOriginShape(GeShape(WeightoriDims));
  weight_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmBatchMatMulDesc->AddInputDesc("w", weight_tensor_desc);

  // add matmul output
  outputy_dims = {input_nz_dim + hidden_nz_dim, batch_nz_dim, 16, 16};
  GeShape outputy_origin_shape(outputy_dims);
  GeShape outputy_shape(outputy_dims);
  GeTensorDesc outputy_tensor_desc = GeTensorDesc(outputy_shape, FORMAT_FRACTAL_NZ, DT_FLOAT);
  outputy_tensor_desc.SetOriginShape(outputy_origin_shape);
  outputy_tensor_desc.SetOriginFormat(FORMAT_FRACTAL_NZ);
  lstmBatchMatMulDesc->AddOutputDesc("y", outputy_tensor_desc);
  // attr
  AttrUtils::SetBool(lstmBatchMatMulDesc, "adj_x1", false);
  AttrUtils::SetBool(lstmBatchMatMulDesc, "adj_x2", true);
  AttrUtils::SetInt(lstmBatchMatMulDesc, "input_size", input_dim);
  AttrUtils::SetInt(lstmBatchMatMulDesc, "hidden_size", hidden_dim);
}

Status DynamicRNNGradAlignFusionPass::AddEdgeForCell(ge::NodePtr dynamicRNNGradNode,
                                                     vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                     vector<vector<ge::NodePtr>> resultNode, ge::NodePtr lstmSplitC,
                                                     ge::NodePtr lstmSplitDy, ge::NodePtr lstmSplitI,
                                                     ge::NodePtr lstmSplitJ, ge::NodePtr lstmSplitF,
                                                     ge::NodePtr lstmSplitO, ge::NodePtr lstmSplitTanh,
                                                     ge::NodePtr lstmXConcatD, ge::NodePtr& lstmGageConcatD) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start Add Edge for loop cell node.");
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  int64_t num_split_x = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["c"]).GetShape().GetDim(0);
  FUSION_PASS_CHECK(resultNode.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "resultNode is null, fusion failed."),
                    failStatus = true);
  FUSION_PASS_CHECK(
      resultNode.size() != 4,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "resultNode lenght is not there, fusion failed."),
      failStatus = true);
  vector<ge::NodePtr> basic_lstm_cell_state_grad_nodes = resultNode[0];
  vector<ge::NodePtr> matmul_nodes = resultNode[1];
  vector<ge::NodePtr> split_nodes = resultNode[2];
  vector<ge::NodePtr> reshape_nodes = {};

  reshape_nodes = resultNode[3];

  // c dy dh dc i j f o tanct
  for (int64_t i = 0; i < num_split_x; i++) {
    // add cell input edge
    int64_t idx = num_split_x - i - 1;  // t-1  -> 0

    if (i == num_split_x - 1) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(0)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitC->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    } else {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmSplitC->GetOutDataAnchor(idx - 1),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(0)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitC->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    }
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(lstmSplitDy->GetOutDataAnchor(idx),
                                           basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(
            FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
            lstmSplitDy->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
        return FAILED);
    // add edge for dh dc
    if (i == 0) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(9)->GetPeerOutAnchor(),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(2)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitDy->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(10)->GetPeerOutAnchor(),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(3)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitDy->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    } else {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(split_nodes[i - 1]->GetOutDataAnchor(1),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(2)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitDy->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i - 1]->GetOutDataAnchor(1),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(3)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitDy->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    }
    if (tSizeJudge != 1) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmSplitI->GetOutDataAnchor(idx),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(4)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitI->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmSplitJ->GetOutDataAnchor(idx),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(5)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitJ->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmSplitF->GetOutDataAnchor(idx),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(6)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitF->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmSplitO->GetOutDataAnchor(idx),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(7)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitO->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmSplitTanh->GetOutDataAnchor(idx),
                                             basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(8)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmSplitTanh->GetName().c_str(), 0, basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    } else {
      ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["i"])->GetPeerOutAnchor(),
                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(CELL_INPUT_INDEX["i"]));
      ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["j"])->GetPeerOutAnchor(),
                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(CELL_INPUT_INDEX["j"]));
      ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["f"])->GetPeerOutAnchor(),
                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(CELL_INPUT_INDEX["f"]));
      ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["o"])->GetPeerOutAnchor(),
                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(CELL_INPUT_INDEX["o"]));
      ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["tanhct"])->GetPeerOutAnchor(),
                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(CELL_INPUT_INDEX["tanhct"]));
    }

    OP_LOGD(FUSED_OP_TYPE.c_str(), "add matmul input1 edge.");
    // add matmul input edge
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i]->GetOutDataAnchor(0),
                                           matmul_nodes[i]->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(
            FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
            basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0, matmul_nodes[i]->GetName().c_str(), 0),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "add matmul input2 edge.");
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                           matmul_nodes[i]->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                       matmul_nodes[i]->GetName().c_str(), 0, matmul_nodes[i]->GetName().c_str(), 0),
        return FAILED);

    // add split input edge
    OP_LOGD(FUSED_OP_TYPE.c_str(), "add split input edge.");
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(matmul_nodes[i]->GetOutDataAnchor(0), split_nodes[i]->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                       matmul_nodes[i]->GetName().c_str(), 0, split_nodes[i]->GetName().c_str(), 0),
        return FAILED);

    // add lstmInputGrad output
    OP_LOGD(FUSED_OP_TYPE.c_str(), "add lstmInputGrad output.");
    if (tSizeJudge != 1) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(split_nodes[i]->GetOutDataAnchor(0), lstmXConcatD->GetInDataAnchor(idx)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              matmul_nodes[i]->GetName().c_str(), 0, split_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    } else {
      lstmXConcatD = split_nodes[i];
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "add lstmInputGrad output.");

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i]->GetOutDataAnchor(0),
                                           reshape_nodes[i]->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                       matmul_nodes[i]->GetName().c_str(), 0, split_nodes[i]->GetName().c_str(), 0),
        return FAILED);
    if (tSizeJudge != 1) {
      FUSION_PASS_CHECK(
          SUCCESS !=
              ge::GraphUtils::AddEdge(reshape_nodes[i]->GetOutDataAnchor(0), lstmGageConcatD->GetInDataAnchor(idx)),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              matmul_nodes[i]->GetName().c_str(), 0, split_nodes[i]->GetName().c_str(), 0),
          return FAILED);
    } else {
      lstmGageConcatD = reshape_nodes[i];
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "add cell input edge.");
    if (i == num_split_x - 1) {
      if (dynamicRNNGradNode->GetOutDataAnchor(OUTPUT_INDEX["dc_prev"])->GetPeerInDataAnchors().size() > 0) {
        for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(
            OUTPUT_INDEX["dc_prev"])->GetPeerInDataAnchors()) {
          inAnchorPtr->UnlinkAll();
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i]->GetOutDataAnchor(1), inAnchorPtr),
              VECTOR_FUSION_INNER_ERR_REPORT(
                  FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                  basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0,
                  basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
              return FAILED);
        }
      }
      if (dynamicRNNGradNode->GetOutDataAnchor(OUTPUT_INDEX["dh_prev"])->GetPeerInDataAnchors().size() > 0) {
        for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(
            OUTPUT_INDEX["dh_prev"])->GetPeerInDataAnchors()) {
          inAnchorPtr->UnlinkAll();
          FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(split_nodes[i]->GetOutDataAnchor(1), inAnchorPtr),
                            VECTOR_FUSION_INNER_ERR_REPORT(
                                FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                split_nodes[i]->GetName().c_str(), 0, split_nodes[i]->GetName().c_str(), 0),
                            return FAILED);
        }
      }
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add output edge for lstmInput.");
  // add output edge for lstmInput
  if (dynamicRNNGradNode->GetOutDataAnchor(OUTPUT_INDEX["dx"])->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(
        OUTPUT_INDEX["dx"])->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(lstmXConcatD->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(
              FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              lstmXConcatD->GetName().c_str(), 0, lstmXConcatD->GetName().c_str(), 0),
          return FAILED);
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "end add edge.");

  return SUCCESS;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                                vector<ge::NodePtr>& newNodes, bool& failStatus) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "start add loop node for graph.");
  vector<vector<ge::NodePtr>> result_node = AddTLoopNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(result_node.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."),
                    failStatus = true);
  // add split for inputs
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();

  GeTensorDesc split_tensor_desc = CreateTensorDescForSplit(dynamicRNNGradDesc);

  int64_t num_split_x = dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["c"]).GetShape().GetDim(0);
  ge::OpDescPtr lstmSplitCDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add splitVD for c.");
  FUSION_PASS_MAKE_SHARED((lstmSplitCDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDC", "SplitVD")),
                          failStatus = true;
                          return nullptr);

  lstmSplitCDesc = SetDescForSplitVDC(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitCDesc);

  ge::OpDescPtr lstmSplitDyDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add splitVD for dy.");
  FUSION_PASS_MAKE_SHARED((lstmSplitDyDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDdy", "SplitVD")),
                          failStatus = true;
                          return nullptr);
  lstmSplitDyDesc = SetDescForSplitVDdy(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitDyDesc);

  ge::OpDescPtr lstmSplitTanhDesc = nullptr;
  ge::OpDescPtr lstmSplitODesc = nullptr;
  ge::OpDescPtr lstmSplitFDesc = nullptr;
  ge::OpDescPtr lstmSplitJDesc = nullptr;
  ge::OpDescPtr lstmSplitIDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add splitVD for t1.");
  if (tSizeJudge != 1) {
    FUSION_PASS_MAKE_SHARED((lstmSplitIDesc = std::make_shared<ge::OpDesc>(
                                 dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDI", "SplitVD")),
                            failStatus = true;
                            return nullptr);
    lstmSplitIDesc = SetDescForSplitVDI(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitIDesc);

    FUSION_PASS_MAKE_SHARED((lstmSplitJDesc = std::make_shared<ge::OpDesc>(
                                 dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDJ", "SplitVD")),
                            failStatus = true;
                            return nullptr);
    lstmSplitJDesc = SetDescForSplitVDJ(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitJDesc);

    FUSION_PASS_MAKE_SHARED((lstmSplitFDesc = std::make_shared<ge::OpDesc>(
                                 dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDF", "SplitVD")),
                            failStatus = true;
                            return nullptr);
    lstmSplitFDesc = SetDescForSplitVDF(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitFDesc);

    FUSION_PASS_MAKE_SHARED((lstmSplitODesc = std::make_shared<ge::OpDesc>(
                                 dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDO", "SplitVD")),
                            failStatus = true;
                            return nullptr);
    lstmSplitODesc = SetDescForSplitVDO(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitODesc);

    FUSION_PASS_MAKE_SHARED((lstmSplitTanhDesc = std::make_shared<ge::OpDesc>(
                                 dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDTanh", "SplitVD")),
                            failStatus = true;
                            return nullptr);
    lstmSplitTanhDesc = SetDescForSplitVDTanh(dynamicRNNGradDesc, split_tensor_desc, num_split_x, lstmSplitTanhDesc);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "add ConcatD for dx.");
  ge::OpDescPtr lstmXConcatDDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((lstmXConcatDDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "/LstmInputGrad/xConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);
  lstmXConcatDDesc = SetDescForxConcatD(result_node, dynamicRNNGradDesc, num_split_x, lstmXConcatDDesc);

  ge::OpDescPtr lstmGageConcatDDesc = nullptr;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add ConcatD for dgate.");
  FUSION_PASS_MAKE_SHARED((lstmGageConcatDDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "/LstmInputGrad/dgateConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);

  lstmGageConcatDDesc = SetDescForDgateConcatD(result_node, dynamicRNNGradDesc, num_split_x, lstmGageConcatDDesc);

  ge::NodePtr lstmSplitC = graph.AddNode(lstmSplitCDesc);
  FUSION_PASS_CHECK(lstmSplitC == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:lstmSplitC is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(lstmSplitC);

  ge::NodePtr lstmSplitDy = graph.AddNode(lstmSplitDyDesc);
  FUSION_PASS_CHECK(lstmSplitDy == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:lstmSplitDy is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(lstmSplitDy);

  ge::NodePtr lstmSplitI = nullptr;
  ge::NodePtr lstmSplitJ = nullptr;
  ge::NodePtr lstmSplitF = nullptr;
  ge::NodePtr lstmSplitO = nullptr;
  ge::NodePtr lstmSplitTanh = nullptr;
  if (tSizeJudge != 1) {
    lstmSplitI = graph.AddNode(lstmSplitIDesc);
    FUSION_PASS_CHECK(lstmSplitI == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmSplitI is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmSplitI);

    lstmSplitJ = graph.AddNode(lstmSplitJDesc);
    FUSION_PASS_CHECK(lstmSplitJ == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmSplitJ is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmSplitJ);

    lstmSplitF = graph.AddNode(lstmSplitFDesc);
    FUSION_PASS_CHECK(lstmSplitF == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmSplitF is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmSplitF);

    lstmSplitO = graph.AddNode(lstmSplitODesc);
    FUSION_PASS_CHECK(lstmSplitO == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmSplitO is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmSplitO);

    lstmSplitTanh = graph.AddNode(lstmSplitTanhDesc);
    FUSION_PASS_CHECK(lstmSplitTanh == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmSplitTanh is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmSplitTanh);
  }
  ge::NodePtr lstmXConcatD = nullptr;
  if (tSizeJudge != 1) {
    lstmXConcatD = graph.AddNode(lstmXConcatDDesc);
    FUSION_PASS_CHECK(lstmXConcatD == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmXConcatD is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmXConcatD);
  }

  ge::NodePtr lstmGageConcatD = nullptr;
  if (tSizeJudge != 1) {
    lstmGageConcatD = graph.AddNode(lstmGageConcatDDesc);
    FUSION_PASS_CHECK(lstmGageConcatD == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode:lstmGageConcatD is null, fusion failed."),
                      failStatus = true);
    newNodes.push_back(lstmGageConcatD);
  }
  // add c
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add Edge for split node.");
  AddEdgeForSplitNode(dynamicRNNGradNode, lstmSplitC, lstmSplitDy, lstmSplitI, lstmSplitJ, lstmSplitF, lstmSplitO,
                      lstmSplitTanh);

  // add edge for cell
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add Edge for loop cell node.");
  AddEdgeForCell(dynamicRNNGradNode, newNodes, failStatus, result_node, lstmSplitC, lstmSplitDy, lstmSplitI,
                 lstmSplitJ, lstmSplitF, lstmSplitO, lstmSplitTanh, lstmXConcatD, lstmGageConcatD);

  return lstmGageConcatD;
}

GeTensorDesc DynamicRNNGradAlignFusionPass::CreateTensorDescForSplit(const OpDescPtr& dynamicRNNGradDesc) const {
  vector<int64_t> splitc_dims;
  splitc_dims.push_back(1);
  splitc_dims.push_back(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["init_c"]).GetShape().GetDim(cIdx0));
  splitc_dims.push_back(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["init_c"]).GetShape().GetDim(cIdx1));
  GeShape splitc_origin_shape(splitc_dims);
  GeShape splitc_shape(splitc_dims);

  GeTensorDesc split_tensor_desc = GeTensorDesc(splitc_shape, FORMAT_ND, DT_FLOAT);
  split_tensor_desc.SetOriginShape(splitc_origin_shape);
  split_tensor_desc.SetOriginFormat(FORMAT_ND);
  return split_tensor_desc;
}

void DynamicRNNGradAlignFusionPass::AddEdgeForSplitNode(const NodePtr& dynamicRNNGradNode, const NodePtr& lstmSplitC,
                                                        const NodePtr& lstmSplitDy, const NodePtr& lstmSplitI,
                                                        const NodePtr& lstmSplitJ, const NodePtr& lstmSplitF,
                                                        const NodePtr& lstmSplitO, const NodePtr& lstmSplitTanh) const {
  GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["c"])->GetPeerOutAnchor(),
                      lstmSplitC->GetInDataAnchor(0));
  GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["dy"])->GetPeerOutAnchor(),
                      lstmSplitDy->GetInDataAnchor(0));
  if (tSizeJudge != 1) {
    GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["i"])->GetPeerOutAnchor(),
                        lstmSplitI->GetInDataAnchor(0));
    GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["j"])->GetPeerOutAnchor(),
                        lstmSplitJ->GetInDataAnchor(0));
    GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["f"])->GetPeerOutAnchor(),
                        lstmSplitF->GetInDataAnchor(0));
    GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["o"])->GetPeerOutAnchor(),
                        lstmSplitO->GetInDataAnchor(0));
    GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["tanhct"])->GetPeerOutAnchor(),
                        lstmSplitTanh->GetInDataAnchor(0));
  }
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForxConcatD(
    const vector<vector<ge::NodePtr>>& result_node, const OpDescPtr& dynamicRNNGradDesc, int64_t num_split_x,
    OpDescPtr& lstmXConcatDDesc) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "add concat for output.");
  vector<NodePtr> split_node = result_node[2];
  GeTensorDesc concat_x_input_tensor_desc =
      GeTensorDesc(split_node[0]->GetOpDesc()->GetOutputDesc(0).GetShape(), FORMAT_FRACTAL_NZ, DT_FLOAT);
  concat_x_input_tensor_desc.SetOriginShape(split_node[0]->GetOpDesc()->GetOutputDesc(0).GetShape());
  concat_x_input_tensor_desc.SetOriginFormat(FORMAT_FRACTAL_NZ);
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmXConcatDDesc->AddInputDesc("dx" + to_string(i + 1), concat_x_input_tensor_desc);
  }

  vector<int64_t> split_concat_output_dims{t_dim, input_nz_dim, batch_nz_dim, 16, 16};
  GeShape split_concat_output_shape(split_concat_output_dims);
  GeTensorDesc dx_output_tensor_desc = GeTensorDesc(split_concat_output_shape, FORMAT_FRACTAL_NZ, DT_FLOAT);
  dx_output_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetOutputDesc(OUTPUT_INDEX["dx"]).GetShape());
  dx_output_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmXConcatDDesc->AddOutputDesc("dx", dx_output_tensor_desc);
  AttrUtils::SetInt(lstmXConcatDDesc, "concat_dim", 0);
  AttrUtils::SetInt(lstmXConcatDDesc, "N", num_split_x);
  return lstmXConcatDDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForDgateConcatD(const vector<vector<ge::NodePtr>>& result_node,
                                                                 const OpDescPtr& dynamicRNNGradDesc,
                                                                 int64_t num_split_x,
                                                                 OpDescPtr& lstmGageConcatDDesc) const {
  vector<NodePtr> matmul_node = result_node[1];
  vector<int64_t> dgate_concat_dims{1, 4 * hidden_nz_dim, batch_nz_dim, 16, 16};
  GeShape dgate_concat_shape(dgate_concat_dims);
  GeTensorDesc concat_gate_input_tensor_desc = GeTensorDesc(dgate_concat_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  vector<int64_t> dgate_concat_ori_dims = matmul_node[0]->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  dgate_concat_ori_dims = {1, dgate_concat_ori_dims[0], dgate_concat_ori_dims[1]};
  concat_gate_input_tensor_desc.SetOriginShape(GeShape(dgate_concat_ori_dims));
  concat_gate_input_tensor_desc.SetOriginFormat(FORMAT_ND);

  for (int64_t i = 0; i < num_split_x; i++) {
    lstmGageConcatDDesc->AddInputDesc("dgate" + to_string(i + 1), concat_gate_input_tensor_desc);
  }
  vector<int64_t> output_dgate_dims{t_dim, batch_dim, 4 * hidden_nz_dim * 16};
  GeShape output_dgate_origin_shape(output_dgate_dims);
  vector<int64_t> output_dgate_nz_dims{t_dim, 4 * hidden_nz_dim, batch_nz_dim, 16, 16};
  GeShape output_dgate_shape(output_dgate_nz_dims);

  GeTensorDesc output_dgate_tensor_desc = GeTensorDesc(output_dgate_shape, FORMAT_FRACTAL_NZ, DT_FLOAT16);
  output_dgate_tensor_desc.SetOriginShape(output_dgate_origin_shape);
  output_dgate_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmGageConcatDDesc->AddOutputDesc("dgate", output_dgate_tensor_desc);
  AttrUtils::SetInt(lstmGageConcatDDesc, "concat_dim", 0);
  AttrUtils::SetInt(lstmGageConcatDDesc, "N", num_split_x);
  return lstmGageConcatDDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDdy(const OpDescPtr& dynamicRNNGradDesc,
                                                              const GeTensorDesc& split_tensor_desc,
                                                              int64_t num_split_x, OpDescPtr& lstmSplitDyDesc) const {
  GeTensorDesc dy_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dy"]).GetShape(), FORMAT_ND, DT_FLOAT);
  dy_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["dy"]).GetShape());
  dy_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitDyDesc->AddInputDesc("dy", dy_input_tensor_desc);
  vector<int64_t> size_splits_dy = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitDyDesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splits_dy.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitDyDesc, "size_splits", size_splits_dy);
  AttrUtils::SetInt(lstmSplitDyDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitDyDesc, "num_split", num_split_x);
  return lstmSplitDyDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDC(const OpDescPtr& dynamicRNNGradDesc,
                                                             const GeTensorDesc& split_tensor_desc, int64_t num_split_x,
                                                             OpDescPtr& lstmSplitCDesc) const {
  GeTensorDesc c_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["c"]).GetShape(), FORMAT_ND, DT_FLOAT);
  c_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["c"]).GetShape());
  c_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitCDesc->AddInputDesc("c", c_input_tensor_desc);
  vector<int64_t> size_splitsc = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitCDesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splitsc.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitCDesc, "size_splits", size_splitsc);
  AttrUtils::SetInt(lstmSplitCDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitCDesc, "num_split", num_split_x);
  return lstmSplitCDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDTanh(const OpDescPtr& dynamicRNNGradDesc,
                                                                const GeTensorDesc& split_tensor_desc,
                                                                int64_t num_split_x,
                                                                OpDescPtr& lstmSplitTanhDesc) const {
  GeTensorDesc tan_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["tanhct"]).GetShape(), FORMAT_ND, DT_FLOAT);
  tan_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["tanhct"]).GetShape());
  tan_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitTanhDesc->AddInputDesc("Tanh", tan_input_tensor_desc);
  vector<int64_t> size_splits_tanh = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitTanhDesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splits_tanh.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitTanhDesc, "size_splits", size_splits_tanh);
  AttrUtils::SetInt(lstmSplitTanhDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitTanhDesc, "num_split", num_split_x);
  return lstmSplitTanhDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDO(const OpDescPtr& dynamicRNNGradDesc,
                                                             const GeTensorDesc& split_tensor_desc, int64_t num_split_x,
                                                             OpDescPtr& lstmSplitODesc) const {
  GeTensorDesc o_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["o"]).GetShape(), FORMAT_ND, DT_FLOAT);
  o_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["o"]).GetShape());
  o_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitODesc->AddInputDesc("O", o_input_tensor_desc);
  vector<int64_t> size_splits_o = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitODesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splits_o.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitODesc, "size_splits", size_splits_o);
  AttrUtils::SetInt(lstmSplitODesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitODesc, "num_split", num_split_x);
  return lstmSplitODesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDF(const OpDescPtr& dynamicRNNGradDesc,
                                                             const GeTensorDesc& split_tensor_desc, int64_t num_split_x,
                                                             OpDescPtr& lstmSplitFDesc) const {
  GeTensorDesc f_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["f"]).GetShape(), FORMAT_ND, DT_FLOAT);
  f_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["f"]).GetShape());
  f_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitFDesc->AddInputDesc("F", f_input_tensor_desc);
  vector<int64_t> size_splits_f = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitFDesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splits_f.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitFDesc, "size_splits", size_splits_f);
  AttrUtils::SetInt(lstmSplitFDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitFDesc, "num_split", num_split_x);
  return lstmSplitFDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDJ(const OpDescPtr& dynamicRNNGradDesc,
                                                             const GeTensorDesc& split_tensor_desc, int64_t num_split_x,
                                                             OpDescPtr& lstmSplitJDesc) const {
  GeTensorDesc j_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["j"]).GetShape(), FORMAT_ND, DT_FLOAT);
  j_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["j"]).GetShape());
  j_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitJDesc->AddInputDesc("J", j_input_tensor_desc);
  vector<int64_t> size_splits_j = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitJDesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splits_j.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitJDesc, "size_splits", size_splits_j);
  AttrUtils::SetInt(lstmSplitJDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitJDesc, "num_split", num_split_x);
  return lstmSplitJDesc;
}

OpDescPtr& DynamicRNNGradAlignFusionPass::SetDescForSplitVDI(const OpDescPtr& dynamicRNNGradDesc,
                                                             const GeTensorDesc& split_tensor_desc, int64_t num_split_x,
                                                             OpDescPtr& lstmSplitIDesc) const {
  GeTensorDesc i_input_tensor_desc =
      GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["i"]).GetShape(), FORMAT_ND, DT_FLOAT);
  i_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(INPUT_INDEX["i"]).GetShape());
  i_input_tensor_desc.SetOriginFormat(FORMAT_ND);
  lstmSplitIDesc->AddInputDesc("I", i_input_tensor_desc);
  vector<int64_t> size_splits_i = {};
  for (int64_t i = 0; i < num_split_x; i++) {
    lstmSplitIDesc->AddOutputDesc("split_c" + to_string(i + 1), split_tensor_desc);
    size_splits_i.push_back(1);
  }
  AttrUtils::SetListInt(lstmSplitIDesc, "size_splits", size_splits_i);
  AttrUtils::SetInt(lstmSplitIDesc, "split_dim", 0);
  AttrUtils::SetInt(lstmSplitIDesc, "num_split", num_split_x);
  return lstmSplitIDesc;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                        vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr splitDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((splitDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/SplitVD", "SplitVD")),
                          failStatus = true;
                          return nullptr);
  // input
  ge::GeTensorDesc inputTensorDescH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]);  // h

  inputTensorDescH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]);
  splitDesc->AddInputDesc("input_h", inputTensorDescH);
  vector<int64_t> outputDims;

  outputDims.push_back(inputTensorDescH.GetShape().GetDim(0) - 1);
  outputDims.push_back(inputTensorDescH.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescH.GetShape().GetDim(2));
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescH.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  splitDesc->AddOutputDesc("split_t_1", outputTensorDesc);

  vector<int64_t> outputLastDims;
  outputLastDims.push_back(1);
  outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(1));
  outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(2));

  ge::GeShape outputLastShape(outputLastDims);
  ge::GeTensorDesc outputLastTensorDesc =
      ge::GeTensorDesc(outputLastShape, ge::FORMAT_ND, inputTensorDescH.GetDataType());
  outputTensorDesc.SetOriginShape(outputLastShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  splitDesc->AddOutputDesc("split_1", outputLastTensorDesc);

  // attr
  vector<int64_t> size_splits;
  size_splits.push_back(inputTensorDescH.GetShape().GetDim(0) - 1);
  size_splits.push_back(1);

  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);

  // create concat node
  ge::NodePtr splitNode = graph.AddNode(splitDesc);
  FUSION_PASS_CHECK(splitNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:splitNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(splitNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["h"])->GetPeerOutAnchor(),
                          splitNode->GetInDataAnchor(0));  // h

  return splitNode;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode,
                                                          ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                          bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/HConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);

  // input
  ge::GeTensorDesc inputTensorDescInitH =
      dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]);  // init_h
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();

  vector<int64_t> input_h;
  input_h.push_back(1);
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(0));
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);

  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  concatDesc->AddInputDesc("input_split_h", inputTensorDescSplitH);

  // output shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDims;

  outputDims.push_back(inputTensorDescSplitH.GetShape().GetDim(0) + 1);
  outputDims.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescInitH.GetShape().GetDim(2));

  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescInitH.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_h", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:concatNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // Init_h
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);
  // input x
  vector<int64_t> input_x_nz_dims{t_dim, input_nz_dim, batch_nz_dim, 16, 16};
  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).Clone();
  inputTensorDescX.SetShape(GeShape(input_x_nz_dims));
  inputTensorDescX.SetFormat(ge::FORMAT_FRACTAL_NZ);
  concatDesc->AddInputDesc("input_x", inputTensorDescX);
  // input init_h
  vector<int64_t> input_inith_nz_dims{t_dim, hidden_nz_dim, batch_nz_dim, 16, 16};
  vector<int64_t> input_inith_dims{t_dim, batch_dim, hidden_dim};
  ge::GeTensorDesc inputTensorDescInitH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  ;
  inputTensorDescInitH.SetShape(GeShape(input_inith_nz_dims));
  inputTensorDescInitH.SetFormat(ge::FORMAT_FRACTAL_NZ);
  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);

  // output
  vector<int64_t> outputDims{t_dim, input_nz_dim + hidden_nz_dim, batch_nz_dim, 16, 16};
  vector<int64_t> outputOriDims{t_dim, batch_dim, (input_nz_dim + hidden_nz_dim) * 16};
  ge::GeTensorDesc outputTensorDesc =
      ge::GeTensorDesc(GeShape(outputDims), ge::FORMAT_FRACTAL_NZ, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(GeShape(outputOriDims));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_xh", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:concatNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // x
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddConcatNodeT_1(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD")),
                          failStatus = true;
                          return nullptr);

  // input x
  vector<int64_t> input_x_nz_dims{t_dim, input_nz_dim, batch_nz_dim, 16, 16};
  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);
  inputTensorDescX.SetShape(GeShape(input_x_nz_dims));
  inputTensorDescX.SetFormat(ge::FORMAT_FRACTAL_NZ);
  concatDesc->AddInputDesc("input_x", inputTensorDescX);
  // input init_h
  vector<int64_t> input_inith_nz_dims{t_dim, hidden_nz_dim, batch_nz_dim, 16, 16};
  vector<int64_t> input_inith_dims{t_dim, batch_dim, hidden_dim};
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]);
  inputTensorDescInitH.SetShape(GeShape(input_inith_nz_dims));
  inputTensorDescInitH.SetOriginShape(GeShape(input_inith_dims));
  inputTensorDescInitH.SetFormat(ge::FORMAT_FRACTAL_NZ);
  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);

  // output
  vector<int64_t> outputDims{t_dim, input_nz_dim + hidden_nz_dim, batch_nz_dim, 16, 16};
  vector<int64_t> outputOriDims{t_dim, input_nz_dim + hidden_nz_dim, batch_dim};
  ge::GeTensorDesc outputTensorDesc =
      ge::GeTensorDesc(GeShape(outputDims), ge::FORMAT_FRACTAL_NZ, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(GeShape(outputOriDims));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_xh", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:concatNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["x"])->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // x
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(INPUT_INDEX["init_h"])->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(1));
  return concatNode;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode,
                                                         ge::NodePtr lstmInputGradNode, ge::ComputeGraph& graph,
                                                         vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  OP_LOGD(FUSED_OP_TYPE.c_str(), "create matmul node for dw.");
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmulDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/BatchMatmul", "BatchMatMul")),
                          failStatus = true;
                          return nullptr);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "create left tensorDesc for matmulDw.");
  // input
  vector<int64_t> input_xh_ori_dims{t_dim, batch_dim, input_nz_dim * 16 + hidden_nz_dim * 16};
  ge::GeTensorDesc inputTensorDescXh = ge::GeTensorDesc(GeShape(input_xh_ori_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDescXh.SetOriginShape(GeShape(input_xh_ori_dims));
  inputTensorDescXh.SetOriginFormat(ge::FORMAT_ND);
  ge::GeTensorDesc inputTensorDescXhTotal = inputTensorDescXh;
  FUSION_PASS_CHECK(
      dynamicRNNGradNode->GetOpDesc() == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get DynamicRnnGrad desc Failed, fusion failed."),
      failStatus = true);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "create right tensorDesc for matmulDw.");
  vector<int64_t> input_dgate_dims{t_dim, batch_dim, hidden_nz_dim * 16 * 4};
  ge::GeTensorDesc inputTensorDescDgate = ge::GeTensorDesc(GeShape(input_dgate_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(GeShape(input_dgate_dims));
  inputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);
  matmulDesc->AddInputDesc("input_xh", inputTensorDescXhTotal);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "create output tensorDesc for matmulDw.");

  vector<int64_t> outputDims{t_dim, input_nz_dim * 16 + hidden_nz_dim * 16, hidden_nz_dim * 16 * 4};
  ge::GeShape outputOriginShape(outputDims);
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(outputOriginShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  // create matmul node
  ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
  FUSION_PASS_CHECK(matmulNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:matmulNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(matmulNode);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge for matmulDw.");
  // Edge
  ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End create node for matmulDw.");
  return matmulNode;
}

Status DynamicRNNGradAlignFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((reduceSumDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ReduceSumD", "ReduceSumD")),
                          return FAILED);
  vector<int64_t> input_x_dims{t_dim, hidden_nz_dim * 4, input_nz_dim + hidden_nz_dim, 16, 16};
  vector<int64_t> input_x_ori_dims{t_dim, (input_nz_dim + hidden_nz_dim) * 16, hidden_nz_dim * 16 * 4};
  ge::GeTensorDesc inputTensorDescMatmul =
      ge::GeTensorDesc(GeShape(input_x_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescMatmul.SetOriginShape(GeShape(input_x_ori_dims));
  inputTensorDescMatmul.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddInputDesc("input_matmul", inputTensorDescMatmul);

  vector<int64_t> output_y_dims{hidden_nz_dim * 4, input_nz_dim + hidden_nz_dim, 16, 16};
  vector<int64_t> output_y_ori_dims{(input_nz_dim + hidden_nz_dim) * 16, hidden_nz_dim * 16 * 4};

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(GeShape(output_y_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(GeShape(output_y_ori_dims));
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(reduceSumNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:reduceSumNode is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(reduceSumNode);

  // add transpose node for nz to fracal_zn_rnn
  bool failStatus = false;
  ge::NodePtr transposeNode = AddTransposeNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(
      failStatus,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddTransposeNode:check failed, fusion failed."),
      return FAILED);

  // create transdata dw desc
  ge::OpDescPtr transdataDwDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transdataDwDesc = std::make_shared<ge::OpDesc>(
      dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/transdataDw", "TransDataRNN")),
                          return FAILED);
  transdataDwDesc = SetDescForTransdataDw(transdataDwDesc);
  // create trans dw node
  ge::NodePtr transDwNode = graph.AddNode(transdataDwDesc);
  FUSION_PASS_CHECK(transDwNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:transDwNode is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(transDwNode);

  // edge
  ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), reduceSumNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), transposeNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), transDwNode->GetInDataAnchor(0));

  if (dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(transDwNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return SUCCESS;
}

OpDescPtr &DynamicRNNGradAlignFusionPass::SetDescForTransdataDw(OpDescPtr &transdataDwDesc) const {
  // input for transdata
  vector<int64_t> trans_zn_rnn_dims{input_nz_dim + hidden_nz_dim, hidden_nz_dim * 4, 16, 16};
  vector<int64_t> trans_ori_nd_dims{input_dim + hidden_dim, hidden_dim * 4};
  GeTensorDesc transDwInDesc =
      GeTensorDesc(GeShape(trans_zn_rnn_dims), FORMAT_FRACTAL_ZN_RNN, DT_FLOAT16);
  transDwInDesc.SetOriginShape(GeShape(trans_ori_nd_dims));
  transDwInDesc.SetOriginFormat(FORMAT_ND);
  transdataDwDesc->AddInputDesc("trans_src", transDwInDesc);
  // output for tarnsdata
  GeTensorDesc transdwOutDesc =
      GeTensorDesc(GeShape(trans_ori_nd_dims), FORMAT_ND, DT_FLOAT16);
  transdwOutDesc.SetOriginShape(GeShape(trans_ori_nd_dims));
  transdwOutDesc.SetOriginFormat(FORMAT_ND);
  transdataDwDesc->AddOutputDesc("trans_dsc", transdwOutDesc);
  // attr
  AttrUtils::SetStr(transdataDwDesc, "src_format", "FRACTAL_ZN_RNN");
  AttrUtils::SetStr(transdataDwDesc, "dst_format", "ND");
  AttrUtils::SetInt(transdataDwDesc, "input_size", input_dim);
  AttrUtils::SetInt(transdataDwDesc, "hidden_size", hidden_dim);
  return transdataDwDesc;
}

Status DynamicRNNGradAlignFusionPass::AddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr lstmInputGradNode,
                                                         ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                         ge::NodePtr const_one_node) {
  // create reduce_sum desc
  OP_LOGD(FUSED_OP_TYPE.c_str(), "create MatMulV2 node for db.");
  ge::OpDescPtr matmulDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((matmulDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/BatchMatMul", "BatchMatMul")),
                          return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "create left tensor for MatMulV2.");
  vector<int64_t> input_dims{t_dim, 4 * hidden_nz_dim, batch_nz_dim, 16, 16};
  ge::GeShape input_shape(input_dims);

  ge::GeTensorDesc inputTensorDescDgate = ge::GeTensorDesc(input_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape());
  inputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "create const tensor for MatMulV2.");
  vector<int64_t> const_dims{t_dim, batch_nz_dim, 1, 16, 16};
  ge::GeShape const_shape(const_dims);
  vector<int64_t> const_origin_dims{t_dim, 1, batch_dim};
  ge::GeShape const_origin_shape(const_origin_dims);
  // input
  ge::GeTensorDesc constDesc = ge::GeTensorDesc(const_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  constDesc.SetOriginShape(const_origin_shape);
  constDesc.SetOriginFormat(ge::FORMAT_ND);

  matmulDesc->AddInputDesc("input_const", constDesc);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "create output tensor for MatMulV2.");

  vector<int64_t> output_dims{t_dim, 4 * hidden_nz_dim, 1, 16, 16};
  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);

  vector<int64_t> output_origin_dims{t_dim, 1, 4 * hidden_nz_dim * 16};
  ge::GeShape output_origin_shape(output_origin_dims);
  outputTensorDesc.SetOriginShape(output_origin_shape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x1", false);
  ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "create reduceSumD for db.");

  ge::OpDescPtr reduceSumDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((reduceSumDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/ReduceSumD", "ReduceSumD")),
                          return FAILED);

  vector<int64_t> reduce_input_dims{t_dim, 1, 4 * hidden_nz_dim * 16};
  ge::GeTensorDesc reducesum_input_desc = ge::GeTensorDesc(GeShape(reduce_input_dims), ge::FORMAT_ND, ge::DT_FLOAT16);
  reducesum_input_desc.SetOriginShape(GeShape(reduce_input_dims));
  reducesum_input_desc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddInputDesc("input_dgate", reducesum_input_desc);

  vector<int64_t> reduce_output_dims{hidden_nz_dim * 4 * 16};
  vector<int64_t> reduce_output_origin_dims{hidden_dim * 4};
  ge::GeTensorDesc outputTensorDescDgate =
      ge::GeTensorDesc(GeShape(reduce_output_dims), ge::FORMAT_ND_RNN_BIAS, ge::DT_FLOAT16);
  outputTensorDescDgate.SetOriginShape(GeShape(reduce_output_origin_dims));
  outputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDescDgate);

  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0, 1});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);
  ge::AttrUtils::SetInt(reduceSumDesc, "hidden_size", hidden_dim);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "create  Transdata RNN_BIAS for db.");

  // create transdata db desc
  ge::OpDescPtr transdataDbDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transdataDbDesc = std::make_shared<ge::OpDesc>(
      dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/transdataDb", "TransDataRNN")), return FAILED);
  transdataDbDesc = SetDescForTransdataDb(transdataDbDesc);

  // create matmul reduce_sum transdata node
  ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
  FUSION_PASS_CHECK(matmulNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:matmulNode is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(matmulNode);

  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(reduceSumNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:reduceSumNode is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(reduceSumNode);

  ge::NodePtr transDataDbNode = graph.AddNode(transdataDbDesc);
  FUSION_PASS_CHECK(transDataDbNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:transDataDbNode is null, fusion failed."),
                    return FAILED);
  newNodes.push_back(transDataDbNode);

  // Edge
  ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const_one_node->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), reduceSumNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), transDataDbNode->GetInDataAnchor(0));
  if (dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(transDataDbNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "end create reduceSumD for db.");
  return SUCCESS;
}

OpDescPtr &DynamicRNNGradAlignFusionPass::SetDescForTransdataDb(OpDescPtr &transdataDbDesc) const {
  // input for transdata
  vector<int64_t> trans_dims{hidden_nz_dim * 4 * 16};
  vector<int64_t> trans_origin_dims{hidden_dim * 4};
  GeTensorDesc transDbInputDesc =
      GeTensorDesc(GeShape(trans_dims), FORMAT_ND_RNN_BIAS, DT_FLOAT16);
  transDbInputDesc.SetOriginShape(GeShape(trans_origin_dims));
  transDbInputDesc.SetOriginFormat(FORMAT_ND);
  transdataDbDesc->AddInputDesc("trans_src", transDbInputDesc);
  //output for tarnsdata
  GeTensorDesc transDboutputDesc =
      GeTensorDesc(GeShape(trans_origin_dims), FORMAT_ND, DT_FLOAT16);
  transDboutputDesc.SetOriginShape(GeShape(trans_origin_dims));
  transDboutputDesc.SetOriginFormat(FORMAT_ND);
  transdataDbDesc->AddOutputDesc("trans_dsc", transDboutputDesc);

  // attr
  AttrUtils::SetStr(transdataDbDesc, "src_format", "ND_RNN_BIAS");
  AttrUtils::SetStr(transdataDbDesc, "dst_format", "ND");
  AttrUtils::SetInt(transdataDbDesc, "input_size", input_dim);
  AttrUtils::SetInt(transdataDbDesc, "hidden_size", hidden_dim);
  return transdataDbDesc;
}

ge::NodePtr DynamicRNNGradAlignFusionPass::AddTransposeNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                            vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create transposed desc
  ge::OpDescPtr transposeDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
                               dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/transposeD", "TransposeD")),
                          failStatus = true;
                          return nullptr);

  // attr
  vector<int32_t> permValue = {1, 0, 3, 2};
  ge::AttrUtils::SetListInt(transposeDesc, "perm", permValue);
  ge::AttrUtils::SetInt(transposeDesc, "input_size", input_dim);
  ge::AttrUtils::SetInt(transposeDesc, "hidden_size", hidden_dim);

  // input
  vector<int64_t> tran_input_zn_dims{hidden_nz_dim * 4, input_nz_dim + hidden_nz_dim, 16, 16};
  vector<int64_t> tran_input_ori_zn_dims{(input_nz_dim + hidden_nz_dim) * 16, hidden_nz_dim * 4 * 16};
  ge::GeTensorDesc transInputDesc =
      ge::GeTensorDesc(GeShape(tran_input_zn_dims), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  transInputDesc.SetOriginShape(GeShape(tran_input_ori_zn_dims));
  transInputDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddInputDesc("x", transInputDesc);

  // output
  vector<int64_t> tran_output_zn_dims{input_nz_dim + hidden_nz_dim, hidden_nz_dim * 4, 16, 16};
  vector<int64_t> tran_out_ori_zn_dims{input_dim + hidden_dim, hidden_dim * 4};
  ge::GeTensorDesc transOutDesc =
      ge::GeTensorDesc(GeShape(tran_output_zn_dims), ge::FORMAT_FRACTAL_ZN_RNN, ge::DT_FLOAT16);
  transOutDesc.SetOriginShape(GeShape(tran_out_ori_zn_dims));
  transOutDesc.SetOriginFormat(ge::FORMAT_ND);
  transposeDesc->AddOutputDesc("y", transOutDesc);

  // create transpose node
  ge::NodePtr transposeNode = graph.AddNode(transposeDesc);
  FUSION_PASS_CHECK(transposeNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusionNode:transposeNode is null, fusion failed."),
                    failStatus = true);
  newNodes.push_back(transposeNode);

  return transposeNode;
}

Status DynamicRNNGradAlignFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;
  // get dynamicRNNGradNode
  ge::NodePtr dynamicRNNGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(
      dynamicRNNGradNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get DynamicRnnGrad Node Failed, fusion failed."),
      return FAILED);

  if (dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_c"]).GetShape().GetDims().size() == 3) {
    cIdx0 = 1;
    cIdx1 = 2;
  } else {
    cIdx0 = 0;
    cIdx1 = 1;
  }

  tSizeJudge = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).GetShape().GetDim(0);
  if (tSizeJudge == -1) {
    return SUCCESS;
  }
  if (PatternFusionUtil::IsUnknownShape(
      dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["dy"]).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["c"]).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["w"]).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["c"]).GetShape().GetDim(2)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).GetShape().GetDim(2)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dh_prev"]).GetShape().GetDim(1)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dx"]).GetShape().GetDim(2)) ||
      PatternFusionUtil::IsUnknownShape(
          dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(OUTPUT_INDEX["dx"]).GetShape().GetDim(1))) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "DynamicRNNGradAlignFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  input_dim = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).GetShape().GetDim(2);
  hidden_dim = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["init_h"]).GetShape().GetDim(1);
  batch_dim = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["x"]).GetShape().GetDim(1);
  t_dim = tSizeJudge;
  if (hidden_dim % 16 == 0 && input_dim % 16 == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "inputsize or hiddensize is 16 align, will not changed.");
    return NOT_CHANGED;
  }

  input_nz_dim = (input_dim + 15) / 16;
  hidden_nz_dim = (hidden_dim + 15) / 16;
  batch_nz_dim = (batch_dim + 15) / 16;

  // add lstmInputGrad
  OP_LOGI(FUSED_OP_TYPE.c_str(), "start add lstmInputGradNode.");
  ge::NodePtr lstmInputGradNode = AddLSTMInputGradNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(
      failStatus,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddLSTMInputGradNode:check failed, fusion failed."),
      return FAILED);

  int t_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(INPUT_INDEX["h"]).GetShape().GetDim(0);
  // add split
  ge::NodePtr concatNode = nullptr;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "start add splitNode and concat node for h.");
  if (t_size != 1) {
    ge::NodePtr splitNode = AddSplitNode(dynamicRNNGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(
        failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddSplitNode:check failed, fusion failed."),
        return FAILED);
    ge::NodePtr hConcatInputNode = nullptr;
    hConcatInputNode = splitNode;
    // add concat
    ge::NodePtr hConcatNode = AddHConcatNode(dynamicRNNGradNode, hConcatInputNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(
        failStatus,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddHConcatNode:check failed, fusion failed."),
        return FAILED);
    // add concat
    concatNode = AddConcatNode(dynamicRNNGradNode, hConcatNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(
        failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddConcatNode:check failed, fusion failed."),
        return FAILED);
  } else {
    // add concat
    concatNode = AddConcatNodeT_1(dynamicRNNGradNode, graph, newNodes, failStatus);
    FUSION_PASS_CHECK(
        failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddConcatNode:check failed, fusion failed."),
        return FAILED);
  }
  // add matmul
  OP_LOGI(FUSED_OP_TYPE.c_str(), "start add matmul node.");
  ge::NodePtr matmulNode =
      AddMatmulNode(dynamicRNNGradNode, concatNode, lstmInputGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AddMatmulNode:check failed, fusion failed."),
                    return FAILED);
  // add dw reduce_sum
  OP_LOGI(FUSED_OP_TYPE.c_str(), "start add reduceSum node for dw.");
  AddDwReduceSumNode(dynamicRNNGradNode, matmulNode, graph, newNodes);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "add Const node for db.");
  ge::NodePtr const_one_node = GetConstNodeOne(dynamicRNNGradNode, graph, newNodes, failStatus);
  // add db reduce_sum
  OP_LOGI(FUSED_OP_TYPE.c_str(), "add reduceSum node for db.");
  AddDbReduceSumNode(dynamicRNNGradNode, lstmInputGradNode, graph, newNodes, const_one_node);
  // unlink all control input of dynamicRNNGradNode
  if (dynamicRNNGradNode->GetInControlAnchor() != nullptr) {
    dynamicRNNGradNode->GetInControlAnchor()->UnlinkAll();
  }
  // unlink all input of dynamicRNNGradNode
  OP_LOGI(FUSED_OP_TYPE.c_str(), "unlink all input of dynamicRNNGradNode.");
  for (auto inAnchor : dynamicRNNGradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "remove dynamicRNNGradNode from graph.");
  // remove dynamicRNNGradNode from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(dynamicRNNGradNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                                   dynamicRNNGradNode->GetName().c_str()),
                    return FAILED);

  return SUCCESS;
}

REGISTER_PASS("DynamicRNNGradAlignFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNGradAlignFusionPass);
}  // namespace fe
