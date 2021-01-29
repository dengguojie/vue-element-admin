/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Concat&Matmul&Reduce))
 *
 */

#include "dynamic_rnn_grad_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace fe {

static const int BLOCKSIZE = 16;
static const char* FUSED_NODE = "DynamicRNNGrad";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNGrad";

vector<FusionPattern*> DynamicRNNGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr GetConstNodeOne(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, bool& failStatus){
    int64_t t_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(7).GetShape().GetDim(0);
    int64_t n_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(7).GetShape().GetDim(1);

    ge::GeTensorPtr assitPtr = nullptr;
    int64_t matrixSize = (n_size + 15)/16*16*16*t_size;
    int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[matrixSize]());
    auto retMem = memset_s(inputAssit.get(), matrixSize, 1, matrixSize);
    FUSION_PASS_CHECK(retMem != EOK, OP_LOGE("DynamicRnnGrad", "Failed to operate memset_s function."), failStatus=true);
    float* dstConst = inputAssit.get();
    for (int j = 0; j < matrixSize; j++) {
        *(dstConst + j) = 1;
    }

    ge::GeTensorDesc tensorDesc;
    vector<int64_t> assit_dim_info = {};
    vector<int64_t> assit_dim_info_origin = {};
    if ((n_value % 16) != 0) {
      assit_dim_info = {t_size, (n_size + 15) / 16, 1, 16, 16};
      assit_dim_info_origin = {t_size, 1, n_size};
    } else {
      assit_dim_info = {t_size*((n_size + 15) / 16), 1, 16, 16};
      assit_dim_info_origin = {1, n_size*t_size};
    }

    ge::GeShape assit_shape(assit_dim_info);
    ge::GeShape assit_shape_origin(assit_dim_info_origin);

    tensorDesc.SetShape(assit_shape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetFormat(ge::FORMAT_FRACTAL_NZ);
    tensorDesc.SetOriginFormat(ge::FORMAT_ND);
    tensorDesc.SetOriginShape(assit_shape_origin);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);

    assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), matrixSize * sizeof(float));

    ge::OpDescPtr const_opdesc = ge::OpDescUtils::CreateConstOp(assitPtr);
    ge::NodePtr const_node = graph.AddNode(const_opdesc);
    newNodes.push_back(const_node);

    return const_node;
}
vector<vector<ge::NodePtr>> DynamicRNNGradFusionPass::AddTLoopNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  vector<vector<ge::NodePtr>> result = {};
  vector<ge::NodePtr> basicLstm_cell_state_grad_nodes={};
  vector<ge::NodePtr> matmul_nodes={};
  vector<ge::NodePtr> split_nodes={};

  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  int64_t t_size = dynamicRNNGradDesc->GetInputDesc(8).GetShape().GetDim(0);
  // all input
  ge::GeTensorDesc inputC = dynamicRNNGradDesc->GetInputDesc(7);
  ge::GeTensorDesc inputDy = dynamicRNNGradDesc->GetInputDesc(8);
  ge::GeTensorDesc inputDh = dynamicRNNGradDesc->GetInputDesc(9);
  ge::GeTensorDesc inputI = dynamicRNNGradDesc->GetInputDesc(11);
  // add split op

  for (int64_t i = 0; i < t_size; i++) {
     // add state_gate op
     ge::OpDescPtr basicLstmCellStateGradDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradDesc->GetName() + "/LstmInputGrad/BasicLSTMCellCStateGrad"+ std::to_string(i),
                                                                             "BasicLSTMCellCStateGradV2");
     // add state_gate op input
     vector<int64_t> cur_tensor_dims;
     cur_tensor_dims.push_back(1);
     cur_tensor_dims.push_back(inputC.GetShape().GetDim(1));
     cur_tensor_dims.push_back(inputC.GetShape().GetDim(2));
     ge::GeShape cur_tensorc_shape(cur_tensor_dims);
     ge::GeShape cur_tensorc_original_shape(cur_tensor_dims);
     ge::GeTensorDesc cur_tensorc = ge::GeTensorDesc(cur_tensorc_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensorc.SetOriginShape(cur_tensorc_original_shape);
     cur_tensorc.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("c", cur_tensorc);

     vector<int64_t> cur_tensor_dy_dims;
     cur_tensor_dy_dims.push_back(1);
     cur_tensor_dy_dims.push_back(inputDy.GetShape().GetDim(1));
     cur_tensor_dy_dims.push_back(inputDy.GetShape().GetDim(2));
     ge::GeShape cur_tensor_dy_shape(cur_tensor_dy_dims);
     ge::GeShape cur_tensor_dy_original_shape(cur_tensor_dy_dims);
     ge::GeTensorDesc cur_tensor_dy = ge::GeTensorDesc(cur_tensor_dy_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensor_dy.SetOriginShape(cur_tensor_dy_original_shape);
     cur_tensor_dy.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("dy", cur_tensor_dy);

     if (i != 0) {
         vector<int64_t> cur_tensor_dh_dims;
         cur_tensor_dh_dims.push_back(1);
         cur_tensor_dh_dims.push_back(dynamicRNNGradDesc->GetInputDesc(9).GetShape().GetDim(0));
         cur_tensor_dh_dims.push_back(dynamicRNNGradDesc->GetInputDesc(9).GetShape().GetDim(1));
         ge::GeShape cur_tensor_dh_shape(cur_tensor_dh_dims);
         ge::GeTensorDesc dh_input_tensor_desc = ge::GeTensorDesc(cur_tensor_dh_shape, ge::FORMAT_ND, ge::DT_FLOAT);
         dh_input_tensor_desc.SetOriginShape(cur_tensor_dh_shape);
         dh_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
         basicLstmCellStateGradDesc->AddInputDesc("dh", dh_input_tensor_desc);
     } else {
         ge::GeTensorDesc dh_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(9).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
         dh_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(9).GetShape());
         dh_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
         basicLstmCellStateGradDesc->AddInputDesc("dh", dh_input_tensor_desc);
     }

     ge::GeTensorDesc dc_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(10).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
     dc_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(10).GetShape());
     dc_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("dc", dc_input_tensor_desc);

     vector<int64_t> cur_tensori_dims;
     cur_tensori_dims.push_back(1);
     cur_tensori_dims.push_back(inputI.GetShape().GetDim(1));
     cur_tensori_dims.push_back(inputI.GetShape().GetDim(2));
     ge::GeShape cur_tensori_shape(cur_tensori_dims);
     ge::GeTensorDesc cur_tensori = ge::GeTensorDesc(cur_tensori_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensori.SetOriginShape(cur_tensorc_original_shape);
     cur_tensori.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("i", cur_tensori);

     ge::GeTensorDesc cur_tensorj = ge::GeTensorDesc(cur_tensori_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensorj.SetOriginShape(cur_tensorc_original_shape);
     cur_tensorj.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("j", cur_tensorj);

     ge::GeTensorDesc cur_tensorf = ge::GeTensorDesc(cur_tensori_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensorf.SetOriginShape(cur_tensorc_original_shape);
     cur_tensorf.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("f", cur_tensorf);

     ge::GeTensorDesc cur_tensoro = ge::GeTensorDesc(cur_tensori_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensoro.SetOriginShape(cur_tensorc_original_shape);
     cur_tensoro.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("o", cur_tensoro);

     ge::GeTensorDesc cur_tensor_tanhct = ge::GeTensorDesc(cur_tensori_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     cur_tensor_tanhct.SetOriginShape(cur_tensorc_original_shape);
     cur_tensor_tanhct.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddInputDesc("tanhct", cur_tensor_tanhct);

     // add state_gate op output
     ge::GeTensorDesc input_tensor_desc_c = basicLstmCellStateGradDesc->GetInputDesc(3);
     vector<int64_t> output_dims;
     output_dims.push_back(input_tensor_desc_c.GetShape().GetDim(0));
     output_dims.push_back(4 * ((((input_tensor_desc_c.GetShape().GetDim(1)) + 15) / 16) * 16));
     ge::GeShape output_origin_shape(output_dims);
     ge::GeShape output_shape(output_dims);
     ge::GeTensorDesc output_tensor_desc = ge::GeTensorDesc(output_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
     output_tensor_desc.SetOriginShape(output_origin_shape);
     output_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddOutputDesc("dgate", output_tensor_desc);

     vector<int64_t> dc_prev_tensor_dims;
     dc_prev_tensor_dims.push_back(inputI.GetShape().GetDim(1));
     dc_prev_tensor_dims.push_back(inputI.GetShape().GetDim(2));
     ge::GeShape dc_prev_tensor_shape(dc_prev_tensor_dims);
     ge::GeTensorDesc tensor_dc_prew = ge::GeTensorDesc(dc_prev_tensor_shape, ge::FORMAT_ND, ge::DT_FLOAT);
     tensor_dc_prew.SetOriginShape(dc_prev_tensor_shape);
     tensor_dc_prew.SetOriginFormat(ge::FORMAT_ND);
     basicLstmCellStateGradDesc->AddOutputDesc("dc_prev", tensor_dc_prew);
     ge::AttrUtils::SetFloat(basicLstmCellStateGradDesc, "forget_bias", 1.0);
     ge::AttrUtils::SetStr(basicLstmCellStateGradDesc, "activation", "Tanh");
     // add matmul
     ge::OpDescPtr lstmBatchMatMulDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradDesc->GetName() + "/LstmInputGrad/Matmul" + std::to_string(i),
                                                                      "MatMulV2");
     // add matmul input
     ge::GeTensorDesc left_tensor_desc = ge::GeTensorDesc(output_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
     left_tensor_desc.SetOriginShape(output_origin_shape);
     left_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
     lstmBatchMatMulDesc->AddInputDesc("dgate", left_tensor_desc);

     lstmBatchMatMulDesc->AddInputDesc("w", dynamicRNNGradDesc->GetInputDesc(1));
     // add matmul output
     vector<int64_t> outputy_dims;
     outputy_dims.push_back((output_tensor_desc.GetShape().GetDim(0) + 15) / 16 );
     outputy_dims.push_back((dynamicRNNGradDesc->GetInputDesc(1).GetOriginShape().GetDim(0) + 15) / 16);
     outputy_dims.push_back(16);
     outputy_dims.push_back(16);
     ge::GeShape outputy_origin_shape(outputy_dims);
     ge::GeShape outputy_shape(outputy_dims);
     ge::GeTensorDesc outputy_tensor_desc = ge::GeTensorDesc(outputy_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
     outputy_tensor_desc.SetOriginShape(outputy_origin_shape);
     outputy_tensor_desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
     lstmBatchMatMulDesc->AddOutputDesc("y", outputy_tensor_desc);
     // attr
     ge::AttrUtils::SetBool(lstmBatchMatMulDesc, "transpose_x1", false);
     ge::AttrUtils::SetBool(lstmBatchMatMulDesc, "transpose_x2", true);

     //add split op
     ge::OpDescPtr lstmSplitDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVD"+ std::to_string(i), "SplitVD");

     vector<int64_t> input_split_dims;
     input_split_dims.push_back((dynamicRNNGradDesc->GetInputDesc(1).GetOriginShape().GetDim(0) + 15)/16);
     input_split_dims.push_back((output_tensor_desc.GetShape().GetDim(0) + 15)/16);
     input_split_dims.push_back(16);
     input_split_dims.push_back(16);
     ge::GeShape input_split_origin_shape(outputy_dims);
     ge::GeShape input_split_shape(input_split_dims);
     ge::GeTensorDesc split_input_tensor_desc = ge::GeTensorDesc(input_split_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
     split_input_tensor_desc.SetOriginShape(input_split_shape);
     split_input_tensor_desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);

     lstmSplitDesc->AddInputDesc("y", split_input_tensor_desc);

     vector<int64_t> dx_dims;
     dx_dims.push_back(1);
     dx_dims.push_back((dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(2) + 15) / 16);
     dx_dims.push_back((dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(1) + 15) / 16);
     dx_dims.push_back(16);
     dx_dims.push_back(16);
     ge::GeShape dx_shape(dx_dims);
     vector<int64_t> dx_ori_dims;
     dx_ori_dims.push_back(1);
     dx_ori_dims.push_back(dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(1));
     dx_ori_dims.push_back(dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(2));
     ge::GeShape dx_original_shape(dx_ori_dims);
     ge::GeTensorDesc tensor_dx = ge::GeTensorDesc(dx_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
     tensor_dx.SetOriginShape(dx_shape);
     tensor_dx.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
     lstmSplitDesc->AddOutputDesc("dx", tensor_dx);

     vector<int64_t> dh_dims;
     dh_dims.push_back(1);
     dh_dims.push_back((dynamicRNNGradDesc->GetOutputDesc(3).GetShape().GetDim(1) + 15) / 16);
     dh_dims.push_back((dynamicRNNGradDesc->GetOutputDesc(3).GetShape().GetDim(0) + 15) / 16);
     dh_dims.push_back(16);
     dh_dims.push_back(16);
     ge::GeShape dh_shape(dh_dims);
     ge::GeTensorDesc dh_tensor_desc = ge::GeTensorDesc(dh_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
     vector<int64_t> dh_ori_dims;
     dh_ori_dims.push_back(1);
     dh_ori_dims.push_back(dynamicRNNGradDesc->GetOutputDesc(3).GetShape().GetDim(0));
     dh_ori_dims.push_back(dynamicRNNGradDesc->GetOutputDesc(3).GetShape().GetDim(1));
     ge::GeShape dh_ori_shape(dh_ori_dims);
     dh_tensor_desc.SetOriginShape(dh_ori_shape);
     dh_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
     lstmSplitDesc->AddOutputDesc("dh_prev", dh_tensor_desc);

     vector<int64_t> size_splits;
     size_splits.push_back((dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(2) + 15) / 16);
     size_splits.push_back((dynamicRNNGradDesc->GetOutputDesc(3).GetShape().GetDim(1) + 15) / 16);
     ge::AttrUtils::SetListInt(lstmSplitDesc, "size_splits", size_splits);
     ge::AttrUtils::SetInt(lstmSplitDesc, "split_dim", 0);
     ge::AttrUtils::SetInt(lstmSplitDesc, "num_split", 2);

     ge::NodePtr basicLstmCellStateGradNode= graph.AddNode(basicLstmCellStateGradDesc);
     FUSION_PASS_CHECK(basicLstmCellStateGradNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", basicLstmCellStateGradDesc->GetName().c_str()), failStatus=true);
     basicLstm_cell_state_grad_nodes.push_back(basicLstmCellStateGradNode);
     newNodes.push_back(basicLstmCellStateGradNode);

     ge::NodePtr matmulNode= graph.AddNode(lstmBatchMatMulDesc);
     FUSION_PASS_CHECK(matmulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmBatchMatMulDesc->GetName().c_str()), failStatus=true);
     matmul_nodes.push_back(matmulNode);
     newNodes.push_back(matmulNode);

     ge::NodePtr splitNode= graph.AddNode(lstmSplitDesc);
     FUSION_PASS_CHECK(splitNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitDesc->GetName().c_str()), failStatus=true);
     split_nodes.push_back(splitNode);
     newNodes.push_back(splitNode);
  }

  result.push_back(basicLstm_cell_state_grad_nodes);
  result.push_back(matmul_nodes);
  result.push_back(split_nodes);

  return result;
}


Status DynamicRNNGradFusionPass::AddEdgeForCell(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                vector<ge::NodePtr>& newNodes, bool& failStatus,
                                                vector<vector<ge::NodePtr>> resultNode, ge::NodePtr lstmSplitC,
                                                ge::NodePtr lstmSplitDy, ge::NodePtr lstmSplitI,
                                                ge::NodePtr lstmSplitJ, ge::NodePtr lstmSplitF,
                                                ge::NodePtr lstmSplitO, ge::NodePtr lstmSplitTanh,
                                                ge::NodePtr lstmXConcatD, ge::NodePtr lstmGageConcatD){
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();
  int64_t num_split_x = dynamicRNNGradDesc->GetInputDesc(7).GetShape().GetDim(0);
  FUSION_PASS_CHECK(resultNode.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "resultNode is null, fusion failed."), failStatus=true);

  vector<ge::NodePtr> basic_lstm_cell_state_grad_nodes = resultNode[0];
  vector<ge::NodePtr> matmul_nodes = resultNode[1];
  vector<ge::NodePtr> split_nodes = resultNode[2];
  // c dy dh dc i j f o tanct
  for (int64_t i = 0; i < num_split_x; i++){
     //add cell input edge
     int64_t idx = num_split_x - i - 1; // t-1  -> 0

     if (i == num_split_x - 1) {
         FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(5)->GetPeerOutAnchor(),
                                                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(0)),
                           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitC->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                           return FAILED);
     } else {
         FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitC->GetOutDataAnchor(idx-1),
                                                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(0)),
                           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitC->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                           return FAILED);
     }
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitDy->GetOutDataAnchor(idx),
                                                          basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(1)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitDy->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     // add edge for dh dc
     if (i == 0) {
         FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(9)->GetPeerOutAnchor(),
                                                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(2)),
                           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitDy->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                           return FAILED);
         FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(10)->GetPeerOutAnchor(),
                                                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(3)),
                           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitDy->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                           return FAILED);

     } else {
         FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(split_nodes[i-1]->GetOutDataAnchor(1),
                                                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(2)),
                           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitDy->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                           return FAILED);
         FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i-1]->GetOutDataAnchor(1),
                                                              basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(3)),
                           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitDy->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                           return FAILED);
     }

     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitI->GetOutDataAnchor(idx),
                                                          basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(4)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitI->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitJ->GetOutDataAnchor(idx),
                                                          basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(5)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitJ->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitF->GetOutDataAnchor(idx),
                                                          basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(6)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitF->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitO->GetOutDataAnchor(idx),
                                                          basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(7)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitO->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmSplitTanh->GetOutDataAnchor(idx),
                                                          basic_lstm_cell_state_grad_nodes[i]->GetInDataAnchor(8)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", lstmSplitTanh->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                       return FAILED);

     //add matmul input edge
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i]->GetOutDataAnchor(0),
                                                          matmul_nodes[i]->GetInDataAnchor(0)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0,matmul_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                                          matmul_nodes[i]->GetInDataAnchor(1)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", matmul_nodes[i]->GetName().c_str(), 0,matmul_nodes[i]->GetName().c_str(), 0),
                       return FAILED);

     //add split input edge
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(matmul_nodes[i]->GetOutDataAnchor(0),
                                                          split_nodes[i]->GetInDataAnchor(0)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", matmul_nodes[i]->GetName().c_str(), 0,split_nodes[i]->GetName().c_str(), 0),
                       return FAILED);

     //add lstmInputGrad output
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(split_nodes[i]->GetOutDataAnchor(0),
                                                          lstmXConcatD->GetInDataAnchor(idx)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", matmul_nodes[i]->GetName().c_str(), 0,split_nodes[i]->GetName().c_str(), 0),
                       return FAILED);
     FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i]->GetOutDataAnchor(0),
                                                          lstmGageConcatD->GetInDataAnchor(idx)),
                       OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.", matmul_nodes[i]->GetName().c_str(), 0,split_nodes[i]->GetName().c_str(), 0),
                       return FAILED);

     if (i == num_split_x - 1) {
         if (dynamicRNNGradNode->GetOutDataAnchor(4)->GetPeerInDataAnchors().size() > 0) {
              for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(4)->GetPeerInDataAnchors()) {
                  inAnchorPtr->UnlinkAll();
                  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(basic_lstm_cell_state_grad_nodes[i]->GetOutDataAnchor(1),
                                                                       inAnchorPtr),
                                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                            basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0,basic_lstm_cell_state_grad_nodes[i]->GetName().c_str(), 0),
                                    return FAILED);
              }
         }
         if (dynamicRNNGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().size() > 0) {
              for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(3)->GetPeerInDataAnchors()) {
                  inAnchorPtr->UnlinkAll();
                  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(split_nodes[i]->GetOutDataAnchor(1),
                                                                       inAnchorPtr),
                                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                            split_nodes[i]->GetName().c_str(), 0, split_nodes[i]->GetName().c_str(), 0),
                                    return FAILED);
              }
         }
     }

  }
  // add output edge for lstmInput
  if (dynamicRNNGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
           inAnchorPtr->UnlinkAll();
           FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(lstmXConcatD->GetOutDataAnchor(0), inAnchorPtr),
                             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                                     lstmXConcatD->GetName().c_str(), 0,lstmXConcatD->GetName().c_str(), 0),
                             return FAILED);

      }
  }

  return SUCCESS;

}
ge::NodePtr DynamicRNNGradFusionPass::AddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                           vector<ge::NodePtr>& newNodes, bool& failStatus) {

  vector<vector<ge::NodePtr>> result_node = AddTLoopNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(result_node.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "result_node is null, fusion failed."), failStatus=true);
  //add split for inputs
  ge::OpDescPtr dynamicRNNGradDesc = dynamicRNNGradNode->GetOpDesc();

  vector<int64_t> splitc_dims;
  splitc_dims.push_back(1);
  splitc_dims.push_back(dynamicRNNGradDesc->GetInputDesc(5).GetShape().GetDim(0));
  splitc_dims.push_back(dynamicRNNGradDesc->GetInputDesc(5).GetShape().GetDim(1));
  ge::GeShape splitc_origin_shape(splitc_dims);
  ge::GeShape splitc_shape(splitc_dims);

  ge::GeTensorDesc split_tensor_desc = ge::GeTensorDesc(splitc_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  split_tensor_desc.SetOriginShape(splitc_origin_shape);
  split_tensor_desc.SetOriginFormat(ge::FORMAT_ND);

  int64_t num_split_x = dynamicRNNGradDesc->GetInputDesc(7).GetShape().GetDim(0);

  ge::OpDescPtr lstmSplitCDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDC", "SplitVD");

  ge::GeTensorDesc c_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(7).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  c_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(7).GetShape());
  c_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitCDesc->AddInputDesc("c", c_input_tensor_desc);
  std::vector<int64_t> size_splitsc= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitCDesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splitsc.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitCDesc, "size_splits", size_splitsc);
  ge::AttrUtils::SetInt(lstmSplitCDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitCDesc, "num_split", num_split_x);

  ge::OpDescPtr lstmSplitDyDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDdy", "SplitVD");
  ge::GeTensorDesc dy_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(8).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  dy_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(8).GetShape());
  dy_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitDyDesc->AddInputDesc("dy", dy_input_tensor_desc);
  std::vector<int64_t> size_splits_dy= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitDyDesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splits_dy.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitDyDesc, "size_splits", size_splits_dy);
  ge::AttrUtils::SetInt(lstmSplitDyDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitDyDesc, "num_split", num_split_x);

  ge::OpDescPtr lstmSplitIDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDI", "SplitVD");
  ge::GeTensorDesc i_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(11).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  i_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(11).GetShape());
  i_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitIDesc->AddInputDesc("I", i_input_tensor_desc);
  std::vector<int64_t> size_splits_i= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitIDesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splits_i.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitIDesc, "size_splits", size_splits_i);
  ge::AttrUtils::SetInt(lstmSplitIDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitIDesc, "num_split", num_split_x);


  ge::OpDescPtr lstmSplitJDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDJ", "SplitVD");
  ge::GeTensorDesc j_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(12).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  j_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(12).GetShape());
  j_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitJDesc->AddInputDesc("J", j_input_tensor_desc);
  std::vector<int64_t> size_splits_j= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitJDesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splits_j.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitJDesc, "size_splits", size_splits_j);
  ge::AttrUtils::SetInt(lstmSplitJDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitJDesc, "num_split", num_split_x);


  ge::OpDescPtr lstmSplitFDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDF", "SplitVD");
  ge::GeTensorDesc f_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(13).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  f_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(13).GetShape());
  f_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitFDesc->AddInputDesc("F", f_input_tensor_desc);
  std::vector<int64_t> size_splits_f= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitFDesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splits_f.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitFDesc, "size_splits", size_splits_f);
  ge::AttrUtils::SetInt(lstmSplitFDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitFDesc, "num_split", num_split_x);

  ge::OpDescPtr lstmSplitODesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDO", "SplitVD");
  ge::GeTensorDesc o_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(14).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  o_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(14).GetShape());
  o_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitODesc->AddInputDesc("O", o_input_tensor_desc);
  std::vector<int64_t> size_splits_o= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitODesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splits_o.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitODesc, "size_splits", size_splits_o);
  ge::AttrUtils::SetInt(lstmSplitODesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitODesc, "num_split", num_split_x);

  ge::OpDescPtr lstmSplitTanhDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/SplitVDTanh", "SplitVD");
  ge::GeTensorDesc tan_input_tensor_desc = ge::GeTensorDesc(dynamicRNNGradDesc->GetInputDesc(15).GetShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tan_input_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetInputDesc(15).GetShape());
  tan_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmSplitTanhDesc->AddInputDesc("Tanh", tan_input_tensor_desc);
  std::vector<int64_t> size_splits_tanh= {};
  for (int64_t i=0; i<num_split_x; i++) {
      lstmSplitTanhDesc->AddOutputDesc("split_c"+ std::to_string(i+1), split_tensor_desc);
      size_splits_tanh.push_back(1);
  }
  ge::AttrUtils::SetListInt(lstmSplitTanhDesc, "size_splits", size_splits_tanh);
  ge::AttrUtils::SetInt(lstmSplitTanhDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(lstmSplitTanhDesc, "num_split", num_split_x);

  // add concat for output
  vector<ge::NodePtr> split_node = result_node[2];
  int64_t split_concat_dim_o = split_node[0]->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1);
  int64_t split_concat_dim1 = split_node[0]->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0);
  vector<int64_t> split_concat_dims;
  split_concat_dims.push_back((split_concat_dim_o + 15)/16);
  split_concat_dims.push_back((split_concat_dim1 + 15)/16);
  split_concat_dims.push_back(16);
  split_concat_dims.push_back(16);
  ge::GeShape split_concat_shape(split_concat_dims);
  ge::GeTensorDesc concat_x_input_tensor_desc = ge::GeTensorDesc(split_node[0]->GetOpDesc()->GetOutputDesc(0).GetShape(), ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  concat_x_input_tensor_desc.SetOriginShape(split_node[0]->GetOpDesc()->GetOutputDesc(0).GetShape());
  concat_x_input_tensor_desc.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);

  ge::OpDescPtr lstmXConcatDDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/xConcatD", "ConcatD");
  for (int64_t i=0; i<num_split_x; i++) {
      lstmXConcatDDesc->AddInputDesc("dx"+ std::to_string(i+1), concat_x_input_tensor_desc);
  }

  vector<int64_t> split_concat_output_dims;
  int64_t split_concat_output_dim_o = dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(2);
  int64_t split_concat_output_dim1 = dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(1);
  split_concat_output_dims.push_back(dynamicRNNGradDesc->GetOutputDesc(2).GetShape().GetDim(0));
  split_concat_output_dims.push_back(((split_concat_output_dim_o + 15)/16));
  split_concat_output_dims.push_back((split_concat_output_dim1 + 15)/16);
  split_concat_output_dims.push_back(16);
  split_concat_output_dims.push_back(16);
  ge::GeShape split_concat_output_shape(split_concat_output_dims);
  ge::GeTensorDesc dx_output_tensor_desc = ge::GeTensorDesc(split_concat_output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
  dx_output_tensor_desc.SetOriginShape(dynamicRNNGradDesc->GetOutputDesc(2).GetShape());
  dx_output_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmXConcatDDesc->AddOutputDesc("dx", dx_output_tensor_desc);
  ge::AttrUtils::SetInt(lstmXConcatDDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(lstmXConcatDDesc, "N", num_split_x);

  vector<ge::NodePtr> matmul_node = result_node[1];
  int64_t dgage_dims0 = matmul_node[0]->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(1);
  int64_t dgage_dims1 = matmul_node[0]->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(0);
  vector<int64_t> dgate_concat_dims;
  int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);
  if ((n_value) % 16 != 0) {
    dgate_concat_dims.push_back(1);
  }
  dgate_concat_dims.push_back((dgage_dims0 + 15)/16);
  dgate_concat_dims.push_back((dgage_dims1 + 15)/16);
  dgate_concat_dims.push_back(16);
  dgate_concat_dims.push_back(16);
  ge::GeShape dgate_concat_shape(dgate_concat_dims);
  ge::GeTensorDesc concat_gate_input_tensor_desc = ge::GeTensorDesc(dgate_concat_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  concat_gate_input_tensor_desc.SetOriginShape(matmul_node[0]->GetOpDesc()->GetInputDesc(0).GetShape());
  concat_gate_input_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  ge::OpDescPtr lstmGageConcatDDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "/LstmInputGrad/dgateConcatD", "ConcatD");
  for (int64_t i=0; i<num_split_x; i++) {
      lstmGageConcatDDesc->AddInputDesc("dgate"+ std::to_string(i+1), concat_gate_input_tensor_desc);
  }
  vector<int64_t> output_dgate_dims;
  ge::GeTensorDesc c_desc = dynamicRNNGradDesc->GetInputDesc(7);
  if ((n_value) % 16 != 0) {
    output_dgate_dims.push_back(c_desc.GetShape().GetDim(0));
    output_dgate_dims.push_back(c_desc.GetShape().GetDim(1));
  }
  else{
    output_dgate_dims.push_back(c_desc.GetShape().GetDim(1) * c_desc.GetShape().GetDim(0));
  }
  output_dgate_dims.push_back(4 * c_desc.GetShape().GetDim(2));
  ge::GeShape output_dgate_origin_shape(output_dgate_dims);

  vector<int64_t> output_dgate_nz_dims;
  if ((n_value) % 16 != 0) {
    output_dgate_nz_dims.push_back(c_desc.GetShape().GetDim(0));
    output_dgate_nz_dims.push_back((dgage_dims0 + 15) / 16);
    output_dgate_nz_dims.push_back((dgage_dims1 + 15) / 16);
  } else {
    output_dgate_nz_dims.push_back((dgage_dims0 + 15) / 16);
    output_dgate_nz_dims.push_back(((dgage_dims1 + 15) / 16) * c_desc.GetShape().GetDim(0));
  }
  output_dgate_nz_dims.push_back(16);
  output_dgate_nz_dims.push_back(16);
  ge::GeShape output_dgate_shape(output_dgate_nz_dims);

  ge::GeTensorDesc output_dgate_tensor_desc = ge::GeTensorDesc(output_dgate_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  output_dgate_tensor_desc.SetOriginShape(output_dgate_origin_shape);
  output_dgate_tensor_desc.SetOriginFormat(ge::FORMAT_ND);
  lstmGageConcatDDesc->AddOutputDesc("dgate", output_dgate_tensor_desc);
  if ((n_value) % 16 != 0) {
    ge::AttrUtils::SetInt(lstmGageConcatDDesc, "concat_dim", 0);
  } else {
    ge::AttrUtils::SetInt(lstmGageConcatDDesc, "concat_dim", 0);
  }
  ge::AttrUtils::SetInt(lstmGageConcatDDesc, "N", num_split_x);

  ge::NodePtr lstmSplitC= graph.AddNode(lstmSplitCDesc);
  FUSION_PASS_CHECK(lstmSplitC == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitCDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitC);

  ge::NodePtr lstmSplitDy= graph.AddNode(lstmSplitDyDesc);
  FUSION_PASS_CHECK(lstmSplitDy == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitDyDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitDy);

  ge::NodePtr lstmSplitI= graph.AddNode(lstmSplitIDesc);
  FUSION_PASS_CHECK(lstmSplitI == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitIDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitI);

  ge::NodePtr lstmSplitJ= graph.AddNode(lstmSplitJDesc);
  FUSION_PASS_CHECK(lstmSplitJ == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitJDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitJ);

  ge::NodePtr lstmSplitF= graph.AddNode(lstmSplitFDesc);
  FUSION_PASS_CHECK(lstmSplitF == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitFDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitF);

  ge::NodePtr lstmSplitO= graph.AddNode(lstmSplitODesc);
  FUSION_PASS_CHECK(lstmSplitO == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitODesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitO);

  ge::NodePtr lstmSplitTanh= graph.AddNode(lstmSplitTanhDesc);
  FUSION_PASS_CHECK(lstmSplitTanh == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmSplitTanhDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmSplitTanh);

  ge::NodePtr lstmXConcatD= graph.AddNode(lstmXConcatDDesc);
  FUSION_PASS_CHECK(lstmXConcatD == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmXConcatDDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmXConcatD);

  ge::NodePtr lstmGageConcatD= graph.AddNode(lstmGageConcatDDesc);
  FUSION_PASS_CHECK(lstmGageConcatD == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", lstmGageConcatDDesc->GetName().c_str()), failStatus=true);
  newNodes.push_back(lstmGageConcatD);

    // add c
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(7)->GetPeerOutAnchor(),
                          lstmSplitC->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(8)->GetPeerOutAnchor(),
                          lstmSplitDy->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(11)->GetPeerOutAnchor(),
                          lstmSplitI->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(12)->GetPeerOutAnchor(),
                          lstmSplitJ->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(13)->GetPeerOutAnchor(),
                          lstmSplitF->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(14)->GetPeerOutAnchor(),
                          lstmSplitO->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(15)->GetPeerOutAnchor(),
                          lstmSplitTanh->GetInDataAnchor(0));

  //add edge for cell
  AddEdgeForCell(dynamicRNNGradNode, graph, newNodes, failStatus,
                 result_node, lstmSplitC, lstmSplitDy, lstmSplitI,
                 lstmSplitJ,  lstmSplitF, lstmSplitO, lstmSplitTanh,
                 lstmXConcatD, lstmGageConcatD);

  return lstmGageConcatD;
}

ge::NodePtr DynamicRNNGradFusionPass::AddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                                   vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create concat desc
  ge::OpDescPtr splitDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/SplitVD", "SplitVD");
  int64_t n_value = 2;
  // input
  ge::GeTensorDesc inputTensorDescH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6);  // h
  if ((n_value % 16) != 0) {
    inputTensorDescH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6);
  } else {
    vector<int64_t> inputhDims;
    inputhDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0) * dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
    inputhDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(2));
    ge::GeShape inputhShape(inputhDims);
    inputTensorDescH = ge::GeTensorDesc(inputhShape, ge::FORMAT_ND, dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetDataType());
    inputTensorDescH.SetOriginShape(inputhShape);
    inputTensorDescH.SetOriginFormat(ge::FORMAT_ND);
  }
  splitDesc->AddInputDesc("input_h", inputTensorDescH);
  vector<int64_t> outputDims;
  if ((n_value % 16) != 0) {
    outputDims.push_back(inputTensorDescH.GetShape().GetDim(0) - 1);
    outputDims.push_back(inputTensorDescH.GetShape().GetDim(1));
    outputDims.push_back(inputTensorDescH.GetShape().GetDim(2));
  } else {
    outputDims.push_back((dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0) - 1) * dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
    outputDims.push_back(inputTensorDescH.GetShape().GetDim(1));
  }

  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescH.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  splitDesc->AddOutputDesc("split_t_1", outputTensorDesc);

  vector<int64_t> outputLastDims;
  if ((n_value % 16) != 0) {
    outputLastDims.push_back(1);
    outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(1));
    outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(2));
  } else {
    outputLastDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
    outputLastDims.push_back(inputTensorDescH.GetShape().GetDim(1));
  }
  ge::GeShape outputLastShape(outputLastDims);
  ge::GeTensorDesc outputLastTensorDesc =
      ge::GeTensorDesc(outputLastShape, ge::FORMAT_ND, inputTensorDescH.GetDataType());
  outputTensorDesc.SetOriginShape(outputLastShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  splitDesc->AddOutputDesc("split_1", outputLastTensorDesc);

  // attr
  vector<int64_t> size_splits;
  if ((n_value % 16) != 0) {
    size_splits.push_back(inputTensorDescH.GetShape().GetDim(0) - 1);
    size_splits.push_back(1);
  } else {
    size_splits.push_back((dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0) - 1) * dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
    size_splits.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
  }

  ge::AttrUtils::SetListInt(splitDesc, "size_splits", size_splits);
  ge::AttrUtils::SetInt(splitDesc, "split_dim", 0);
  ge::AttrUtils::SetInt(splitDesc, "num_split", 2);

  // create concat node
  ge::NodePtr splitNode = graph.AddNode(splitDesc);
  FUSION_PASS_CHECK(splitNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                  splitNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(splitNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(6)->GetPeerOutAnchor(),
                          splitNode->GetInDataAnchor(0));  // h

  return splitNode;
}

ge::NodePtr DynamicRNNGradFusionPass::AddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode,
                                                     ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                     bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/HConcatD", "ConcatD");

  // input
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(4);  // init_h
  ge::GeTensorDesc inputTensorDescSplitH = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();

  vector<int64_t> input_h;
  int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);
  if ((n_value % 16) != 0) {
    input_h.push_back(1);
  }
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(0));
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);

  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  concatDesc->AddInputDesc("input_split_h", inputTensorDescSplitH);

  // output shape:{t,batch_size,hidden_size}
  vector<int64_t> outputDims;
  if ((n_value % 16) != 0) {
    outputDims.push_back(inputTensorDescSplitH.GetShape().GetDim(0) + 1);
    outputDims.push_back(inputTensorDescInitH.GetShape().GetDim(1));
    outputDims.push_back(inputTensorDescInitH.GetShape().GetDim(2));
  } else {
    outputDims.push_back((dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0)) * dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
    outputDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(2));
  }

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
  FUSION_PASS_CHECK(concatNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   concatNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // Init_h
  ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}

ge::NodePtr DynamicRNNGradFusionPass::AddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                    bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD");
  int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);

  // input
  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);  // x
  if ((n_value % 16) != 0) {
    inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);
  } else {
    vector<int64_t> inputxDims;
    inputxDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(0) * dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(1));
    inputxDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(2));
    ge::GeShape inputxShape(inputxDims);
    inputTensorDescX = ge::GeTensorDesc(inputxShape, ge::FORMAT_ND, dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetDataType());
    inputTensorDescX.SetOriginShape(inputxShape);
    inputTensorDescX.SetOriginFormat(ge::FORMAT_ND);
  }
  ge::GeTensorDesc inputTensorDescH = hConcatNode->GetOpDesc()->GetOutputDesc(0).Clone();
  concatDesc->AddInputDesc("input_x", inputTensorDescX);
  concatDesc->AddInputDesc("input_h", inputTensorDescH);
  // output shape:{t,batch_size,input_size+hidden_size}
  vector<int64_t> outputDims;
  if ((n_value % 16) != 0) {
    outputDims.push_back(inputTensorDescX.GetShape().GetDim(0));
    outputDims.push_back(inputTensorDescX.GetShape().GetDim(1));
    outputDims.push_back(inputTensorDescX.GetShape().GetDim(2) + inputTensorDescH.GetShape().GetDim(2));
  } else {
    outputDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0) * dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1));
    outputDims.push_back(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(2) + inputTensorDescH.GetShape().GetDim(1));
  }

  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_xh", outputTensorDesc);
  // attr
  if ((n_value % 16) != 0) {
    ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  } else {
    ge::AttrUtils::SetInt(concatDesc, "concat_dim", 1);
  }
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   concatNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // x
  ge::GraphUtils::AddEdge(hConcatNode->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1));

  return concatNode;
}
ge::NodePtr DynamicRNNGradFusionPass::AddConcatNodeT_1(ge::NodePtr dynamicRNNGradNode,
                                                       ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                       bool& failStatus) {
  // create concat desc
  ge::OpDescPtr concatDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ConcatD", "ConcatD");

  // input
  ge::GeTensorDesc inputTensorDescX = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0);  // x
  ge::GeTensorDesc inputTensorDescInitH = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(4);
  concatDesc->AddInputDesc("input_x", inputTensorDescX);

  vector<int64_t> input_h;
  input_h.push_back(1);
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(0));
  input_h.push_back(inputTensorDescInitH.GetShape().GetDim(1));
  ge::GeShape init_hShape(input_h);
  inputTensorDescInitH.SetShape(init_hShape);
  inputTensorDescInitH.SetOriginShape(init_hShape);
  concatDesc->AddInputDesc("input_init_h", inputTensorDescInitH);
  vector<int64_t> outputDims;
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(0));
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(1));
  outputDims.push_back(inputTensorDescX.GetShape().GetDim(2) + inputTensorDescInitH.GetShape().GetDim(2));
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, inputTensorDescX.GetDataType());
  outputTensorDesc.SetOriginShape(outputShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  concatDesc->AddOutputDesc("concat_xh", outputTensorDesc);
  // attr
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 2);
  ge::AttrUtils::SetInt(concatDesc, "N", 2);

  // create concat node
  ge::NodePtr concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   concatNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(concatNode);

  // Edge
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(0));  // x
  ge::GraphUtils::AddEdge(dynamicRNNGradNode->GetInDataAnchor(4)->GetPeerOutAnchor(),
                          concatNode->GetInDataAnchor(1));
  return concatNode;
}
ge::NodePtr DynamicRNNGradFusionPass::AddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode,
                                                    ge::NodePtr lstmInputGradNode, ge::ComputeGraph& graph,
                                                    vector<ge::NodePtr>& newNodes, bool& failStatus) {
  // create matmul desc
  int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);
  ge::OpDescPtr matmulDesc = nullptr;
  if ((n_value) % 16 != 0) {
    matmulDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/BatchMatmul", "BatchMatMul");
  } else {
    matmulDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/BatchMatmul", "MatMulV2");
  }

  // input
  ge::GeTensorDesc inputTensorDescXh = ge::GeTensorDesc(concatNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDescXh.SetOriginShape(concatNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape());
  inputTensorDescXh.SetOriginFormat(ge::FORMAT_ND);

  ge::GeTensorDesc inputTensorDescDgate = ge::GeTensorDesc(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape());
  inputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);
  matmulDesc->AddInputDesc("input_xh", inputTensorDescXh);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);
  vector<int64_t> outputDims;
  if ((n_value % 16) != 0) {
    outputDims.push_back(inputTensorDescXh.GetShape().GetDim(0));
    outputDims.push_back(inputTensorDescXh.GetShape().GetDim(2));
    outputDims.push_back(inputTensorDescDgate.GetOriginShape().GetDim(2));
  } else {
    outputDims.push_back(inputTensorDescXh.GetShape().GetDim(1));
    outputDims.push_back(inputTensorDescDgate.GetOriginShape().GetDim(1));
  }
  ge::GeShape outputOriginShape(outputDims);
  ge::GeShape outputShape(outputDims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(outputShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(outputOriginShape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  if ((n_value) % 16 != 0) {
    ge::AttrUtils::SetBool(matmulDesc, "adj_x1", true);
    ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);
  } else {
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x1", true);
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x2", false);
  }

  // create matmul node
  ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
  FUSION_PASS_CHECK(matmulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                   matmulNode->GetName().c_str()),
                    failStatus = true);
  newNodes.push_back(matmulNode);

  // Edge
  ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));  // dgate

  return matmulNode;
}

Status DynamicRNNGradFusionPass::AddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes) {
  // create reduce_sum desc
  ge::OpDescPtr reduceSumDesc =
      std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Dw/ReduceSumD", "ReduceSumD");
  vector<int64_t> input_dims;
  input_dims.push_back(matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0));
  input_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) + 15) / 16);
  input_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(1) + 15) / 16);
  input_dims.push_back(16);
  input_dims.push_back(16);
  ge::GeShape input_shape(input_dims);
  ge::GeTensorDesc inputTensorDescMatmul = ge::GeTensorDesc(input_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescMatmul.SetOriginShape(matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape());
  inputTensorDescMatmul.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddInputDesc("input_matmul", inputTensorDescMatmul);
  vector<int64_t> output_dims;
  output_dims.push_back(1);
  output_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) + 15) / 16);
  output_dims.push_back((matmulNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(1) + 15) / 16);
  output_dims.push_back(16);
  output_dims.push_back(16);
  ge::GeShape output_shape(output_dims);
  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  outputTensorDesc.SetOriginShape(dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(0).GetShape());
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);
  reduceSumDesc->AddOutputDesc("y", outputTensorDesc);
  // attr
  ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0});
  ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

  // create reduce_sum node
  ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
  FUSION_PASS_CHECK(reduceSumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                      reduceSumNode->GetName().c_str()),
                    return FAILED);
  newNodes.push_back(reduceSumNode);

  // Edge
  ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), reduceSumNode->GetInDataAnchor(0));
  if (dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {  // dw
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return SUCCESS;
}

Status DynamicRNNGradFusionPass::AddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr lstmInputGradNode,
                                                    ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                                    ge::NodePtr const_one_node) {
  // create reduce_sum desc
  int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);
  ge::OpDescPtr matmulDesc = nullptr;
  if ((n_value % 16) != 0) {
    matmulDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/BatchMatMul", "BatchMatMul");
  } else {
    matmulDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/BatchMatMul", "MatMulV2");
  }

  vector<int64_t> input_dims;
  if ((n_value) % 16 != 0) {
    input_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0));
    input_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(2) + 15) / 16);
    input_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1) + 15) / 16);
  } else {
    input_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1) + 15) / 16);
    input_dims.push_back(((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0) + 15) / 16));
  }
  input_dims.push_back(16);
  input_dims.push_back(16);
  ge::GeShape input_shape(input_dims);

  ge::GeTensorDesc inputTensorDescDgate = ge::GeTensorDesc(input_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  inputTensorDescDgate.SetOriginShape(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape());
  inputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

  vector<int64_t> const_dims;
  if ((n_value % 16) != 0) {
    const_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0));
    const_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1) + 15) / 16);
    const_dims.push_back(1);
  } else {
    const_dims.push_back(((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0) + 15) / 16));
    const_dims.push_back(1);
  }
  const_dims.push_back(16);
  const_dims.push_back(16);
  ge::GeShape const_shape(const_dims);

  vector<int64_t> const_origin_dims;
  if ((n_value) % 16 != 0) {
    const_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0));
    const_origin_dims.push_back(1);
    const_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1));
  } else {
    const_origin_dims.push_back(1);
    const_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0));
  }
  ge::GeShape const_origin_shape(const_origin_dims);
  // input
  ge::GeTensorDesc constDesc = ge::GeTensorDesc(const_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
  constDesc.SetOriginShape(const_origin_shape);
  constDesc.SetOriginFormat(ge::FORMAT_ND);

  matmulDesc->AddInputDesc("input_const", constDesc);
  matmulDesc->AddInputDesc("input_dgate", inputTensorDescDgate);

  vector<int64_t> output_dims;
  if ((n_value) % 16 != 0) {
    output_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0));
    output_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(2) + 15) / 16);
  } else {
    output_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1) + 15) / 16);
  }
  output_dims.push_back(1);
  output_dims.push_back(16);
  output_dims.push_back(16);
  ge::GeShape output_shape(output_dims);

  ge::GeTensorDesc outputTensorDesc = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);

  vector<int64_t> output_origin_dims;
  if ((n_value) % 16 != 0) {
    output_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(0));
    output_origin_dims.push_back(1);
    output_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(2));
  } else {
    output_origin_dims.push_back(1);
    output_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(1));
  }

  ge::GeShape output_origin_shape(output_origin_dims);

  outputTensorDesc.SetOriginShape(output_origin_shape);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_ND);

  matmulDesc->AddOutputDesc("y", outputTensorDesc);
  if ((n_value) % 16 != 0) {
    ge::AttrUtils::SetBool(matmulDesc, "adj_x1", false);
    ge::AttrUtils::SetBool(matmulDesc, "adj_x2", false);
  } else {
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x1", false);
    ge::AttrUtils::SetBool(matmulDesc, "transpose_x2", false);
  }
  if ((n_value % 16) != 0) {
    ge::OpDescPtr reduceSumDesc = std::make_shared<ge::OpDesc>(dynamicRNNGradNode->GetName() + "LSTMWeightGrad/Db/ReduceSumD", "ReduceSumD");
    ge::GeTensorDesc reduceInputTensorDescDgate = ge::GeTensorDesc(output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    reduceInputTensorDescDgate.SetOriginShape(output_origin_shape);
    reduceInputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

    reduceSumDesc->AddInputDesc("input_dgate", reduceInputTensorDescDgate);

    vector<int64_t> reduce_output_dims;
    reduce_output_dims.push_back(1);
    reduce_output_dims.push_back((lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(2) + 15) / 16);
    reduce_output_dims.push_back(1);
    reduce_output_dims.push_back(16);
    reduce_output_dims.push_back(16);
    ge::GeShape reduce_output_shape(reduce_output_dims);


    vector<int64_t> reduce_output_origin_dims;
    reduce_output_origin_dims.push_back(1);
    reduce_output_origin_dims.push_back(1);
    reduce_output_origin_dims.push_back(lstmInputGradNode->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDim(2));
    ge::GeShape reduce_output_origin_shape(reduce_output_origin_dims);

    ge::GeTensorDesc outputTensorDescDgate = ge::GeTensorDesc(reduce_output_shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT16);
    outputTensorDescDgate.SetOriginShape(reduce_output_origin_shape);
    outputTensorDescDgate.SetOriginFormat(ge::FORMAT_ND);

    reduceSumDesc->AddOutputDesc("y", outputTensorDescDgate);

    // attr
    ge::AttrUtils::SetListInt(reduceSumDesc, "axes", {0});
    ge::AttrUtils::SetBool(reduceSumDesc, "keep_dims", false);

    // create reduce_sum node
    ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
    FUSION_PASS_CHECK(matmulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                     matmulNode->GetName().c_str()),
                      return FAILED);
    newNodes.push_back(matmulNode);
    ge::NodePtr reduceSumNode = graph.AddNode(reduceSumDesc);
    FUSION_PASS_CHECK(reduceSumNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                                        reduceSumNode->GetName().c_str()),
                      return FAILED);
    newNodes.push_back(reduceSumNode);

    // Edge
    ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(const_one_node->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), reduceSumNode->GetInDataAnchor(0));      // dgate
    if (dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {                        // db
      for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {  // db
        inAnchorPtr->UnlinkAll();
        ge::GraphUtils::AddEdge(reduceSumNode->GetOutDataAnchor(0), inAnchorPtr);
      }
    }
  } else {
    ge::NodePtr matmulNode = graph.AddNode(matmulDesc);
    FUSION_PASS_CHECK(matmulNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddLSTMInputGradNode:check failed, fusion failed."),
                      return FAILED);
    newNodes.push_back(matmulNode);
    ge::GraphUtils::AddEdge(lstmInputGradNode->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(const_one_node->GetOutDataAnchor(0), matmulNode->GetInDataAnchor(0));
    if (dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {                        // db
      for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {  // db
        inAnchorPtr->UnlinkAll();
        ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
      }
    }
  }

  return SUCCESS;
}

Status DynamicRNNGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;
  // get dynamicRNNGradNode
  ge::NodePtr dynamicRNNGradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  if (PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(8).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(7).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(1).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(7).GetShape().GetDim(2)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(2)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(3).GetShape().GetDim(0)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(3).GetShape().GetDim(1)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(2).GetShape().GetDim(2)) ||
      PatternFusionUtil::IsUnknownShape(dynamicRNNGradNode->GetOpDesc()->GetOutputDesc(2).GetShape().GetDim(1))) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "DynamicRNNGradFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  // add lstmInputGrad
  ge::NodePtr lstmInputGradNode = AddLSTMInputGradNode(dynamicRNNGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddLSTMInputGradNode:check failed, fusion failed."),
                    return FAILED);

  int t_size = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(0);
  // add split
  int64_t n_value = dynamicRNNGradNode->GetOpDesc()->GetInputDesc(6).GetShape().GetDim(1);
  ge::NodePtr concatNode = nullptr;
  if (t_size != 1) {
        ge::NodePtr splitNode = AddSplitNode(dynamicRNNGradNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddSplitNode:check failed, fusion failed."),
                          return FAILED);
        ge::NodePtr hConcatInputNode = nullptr;
        if (n_value % 16 == 0) {
          auto reshapeOp = ge::OperatorFactory::CreateOperator(dynamicRNNGradNode->GetName() + "/myReshape", "Reshape");
          FUSION_PASS_CHECK(reshapeOp.IsEmpty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "create Reshape Op operator error."),
                          return FAILED);
          auto reshape_desc = ge::OpDescUtils::GetOpDescFromOperator(reshapeOp);
          reshapeOp.BreakConnect();
          reshape_desc->UpdateInputDesc("x", splitNode->GetOpDesc()->GetOutputDesc(0));

          ge::GeTensorDesc inputTensorDescSplitHClone = splitNode->GetOpDesc()->GetOutputDesc(0).Clone();
          vector<int64_t> inputTensorDescSplitHDims;
          inputTensorDescSplitHDims.push_back(inputTensorDescSplitHClone.GetShape().GetDim(0) * inputTensorDescSplitHClone.GetShape().GetDim(1));
          inputTensorDescSplitHDims.push_back(inputTensorDescSplitHClone.GetShape().GetDim(2));
          ge::GeShape inputTensorDescSplitHShape(inputTensorDescSplitHDims);
          ge::GeTensorDesc inputTensorDescSplitH = ge::GeTensorDesc(inputTensorDescSplitHShape, ge::FORMAT_ND, inputTensorDescSplitHClone.GetDataType());
          inputTensorDescSplitH.SetOriginShape(inputTensorDescSplitHShape);
          inputTensorDescSplitH.SetOriginFormat(ge::FORMAT_ND);
          reshape_desc->UpdateInputDesc("shape", inputTensorDescSplitH);
          reshape_desc->UpdateOutputDesc("y", inputTensorDescSplitH);

          ge::NodePtr myReshape_node = graph.AddNode(reshape_desc);
          FUSION_PASS_CHECK(myReshape_node==nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "create Reshape node  error."),
                            return FAILED);
          newNodes.push_back(myReshape_node);

          hConcatInputNode = myReshape_node;
          ge::GraphUtils::AddEdge(splitNode->GetOutDataAnchor(0), myReshape_node->GetInDataAnchor(0));
        } else {
          hConcatInputNode = splitNode;
        }
        // add concat
        ge::NodePtr hConcatNode = AddHConcatNode(dynamicRNNGradNode, hConcatInputNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddHConcatNode:check failed, fusion failed."),
                          return FAILED);
        // add concat
        concatNode = AddConcatNode(dynamicRNNGradNode, hConcatNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddConcatNode:check failed, fusion failed."),
                          return FAILED);
  } else {
        // add concat
        concatNode = AddConcatNodeT_1(dynamicRNNGradNode, graph, newNodes, failStatus);
        FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddConcatNode:check failed, fusion failed."),
                          return FAILED);

  }
  // add matmul
  ge::NodePtr matmulNode =
      AddMatmulNode(dynamicRNNGradNode, concatNode, lstmInputGradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, OP_LOGE(FUSED_OP_TYPE.c_str(), "AddMatmulNode:check failed, fusion failed."),
                    return FAILED);
  if (t_size != 1 && (n_value % 16) != 0) {
      // add dw reduce_sum
      AddDwReduceSumNode(dynamicRNNGradNode, matmulNode, graph, newNodes);
  } else {
      // Edge
      if (dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
         for (InDataAnchorPtr inAnchorPtr : dynamicRNNGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {  // dw
             inAnchorPtr->UnlinkAll();
             ge::GraphUtils::AddEdge(matmulNode->GetOutDataAnchor(0), inAnchorPtr);
         }
      }
  }
  ge::NodePtr const_one_node = GetConstNodeOne(dynamicRNNGradNode, graph, newNodes, failStatus);
  // add db reduce_sum
  AddDbReduceSumNode(dynamicRNNGradNode, lstmInputGradNode, graph, newNodes, const_one_node);
  // unlink all control input of dynamicRNNGradNode
  if (dynamicRNNGradNode->GetInControlAnchor() != nullptr) {
    dynamicRNNGradNode->GetInControlAnchor()->UnlinkAll();
  }
  // unlink all input of dynamicRNNGradNode
  for (auto inAnchor : dynamicRNNGradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  // remove dynamicRNNGradNode from graph
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(dynamicRNNGradNode),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed", dynamicRNNGradNode->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

REGISTER_PASS("DynamicRNNGradFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNGradFusionPass);
}
