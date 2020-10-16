
#include "unsortedsegmentsumd_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const string PATTERN_UNSORTED_SEGMENT_SUM = "UnsortedSegmentSumD";
static const string PATTERNUNSORTEDSEGMENT_SUM = "UnsortedSegmentSumD";


vector<FusionPattern *> UnsortedSegmentSumdFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new(std::nothrow) FusionPattern("UnsortedSegmentSumd1to8Fusion");

  FUSION_PASS_CHECK(pattern == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_UNSORTED_SEGMENT_SUM, {PATTERNUNSORTEDSEGMENT_SUM})
          .SetOutput(PATTERN_UNSORTED_SEGMENT_SUM);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define UnsortedSegmentSumdFusionPass pattern end");
  return patterns;
}


Status UnsortedSegmentSumdFusionPass::Fusion(ge::ComputeGraph &graph,
                                             Mapping &mapping,
                                             vector<ge::NodePtr> &fusionNodes) {

  ge::NodePtr UnsortedSegmentSumdNode = GetNodeFromMapping(PATTERN_UNSORTED_SEGMENT_SUM, mapping);
  ge::OpDescPtr UnsortedSegmentSumdNodetransDesc = UnsortedSegmentSumdNode->GetOpDesc();
  FUSION_PASS_CHECK(UnsortedSegmentSumdNodetransDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "TransposeNode's OpDesc is null, fusion failed."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(UnsortedSegmentSumdNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumdNode is null, fusion failed."),
           return PARAM_INVALID);

  FUSION_PASS_CHECK(UnsortedSegmentSumdNode->GetInDataNodes().size() != 2,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of UnsortedSegmentSumdNode node size is [%lu], which not equal to 2.",
                   UnsortedSegmentSumdNode->GetInDataNodes().size()),
           return NOT_CHANGED);
  FUSION_PASS_CHECK(UnsortedSegmentSumdNode->GetOutDataNodes().size() != 1,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of UnsortedSegmentSumdNode node size is [%d], which not equal to 1.",
                   UnsortedSegmentSumdNode->GetOutDataNodes().size()),
           return NOT_CHANGED);

  // getshape，ifnot[xxx,1] return
  std::vector<int64_t> secondDims = UnsortedSegmentSumdNodetransDesc->GetInputDesc(0).GetOriginShape().GetDims();
  int64_t secondDimsSize=secondDims.size();
  FUSION_PASS_CHECK(secondDims[secondDimsSize-1] != 1,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumdNode is not need fusion.secondDims[secondDimsSize-1]=[%d]",
                   secondDims[secondDimsSize-1]),
           return NOT_CHANGED);

  //1 copy Opdesc
  std::shared_ptr<ge::OpDesc> AicpuPad1to8fusedNode_desc = nullptr;
  AicpuPad1to8fusedNode_desc =
      std::make_shared<ge::OpDesc>(UnsortedSegmentSumdNode->GetName() + "/" + "AicpuPad1to8First", "AscendPadding");
  FUSION_PASS_CHECK(AicpuPad1to8fusedNode_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "AicpuPad1to8fusedNode_desc is null, fusion failed."),
           return PARAM_INVALID);

  // add input
  ge::GeTensorDesc input_desc = UnsortedSegmentSumdNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(AicpuPad1to8fusedNode_desc->AddInputDesc(input_desc) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);

  // add output
  ge::GeShape output_desc_shape1 = input_desc.GetShape();
  ge::DataType input_desc_type = input_desc.GetDataType();
  int padDimSize;
  if(input_desc_type == DT_FLOAT){
     padDimSize = 8;
  } else if (input_desc_type == DT_FLOAT16) {
     padDimSize = 16;
  } else {
     OP_LOGI(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumdNode dtype is not in (float32,float16),no need change");
     return NOT_CHANGED;
  }
  //output  [16000,39,1] ->[16000,39,8]
  output_desc_shape1.SetDim(secondDimsSize-1, padDimSize);
  ge::Format input_desc_Format = input_desc.GetFormat();
  ge::GeTensorDesc tensorDescPaddingOutput(GeShape(), input_desc_Format, input_desc_type);
  tensorDescPaddingOutput.SetShape(output_desc_shape1);
  tensorDescPaddingOutput.SetOriginFormat(input_desc_Format);
  tensorDescPaddingOutput.SetOriginDataType(input_desc_type);

  FUSION_PASS_CHECK(AicpuPad1to8fusedNode_desc->AddOutputDesc(tensorDescPaddingOutput) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);

  //add AicpuPad1to8fused_node node
  ge::NodePtr AicpuPad1to8fused_node = graph.AddNode(AicpuPad1to8fusedNode_desc);
  fusionNodes.push_back(AicpuPad1to8fused_node);
  ge::AttrUtils::SetInt(AicpuPad1to8fused_node->GetOpDesc(), "pad_dim_size", padDimSize);

  //2 copy Opdesc
  std::shared_ptr<ge::OpDesc> UnsortedSegmentSumd8_desc = nullptr;
  UnsortedSegmentSumd8_desc =
      std::make_shared<ge::OpDesc>(UnsortedSegmentSumdNode->GetName() + "/" + "UnsortedSegmentSumd8", "UnsortedSegmentSumD");
  FUSION_PASS_CHECK(UnsortedSegmentSumd8_desc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumd8_desc is null, fusion failed."),
           return PARAM_INVALID);

  // add input
  ge::GeTensorDesc input_desc0 = AicpuPad1to8fused_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(UnsortedSegmentSumd8_desc->AddInputDesc(input_desc0) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);
  ge::GeTensorDesc input_desc1 = UnsortedSegmentSumdNode->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(UnsortedSegmentSumd8_desc->AddInputDesc(input_desc1) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);

  // add output shape  refence SoftmaxCrossEntropyWithLogitsGradPass SetShape
  //input [xxx,1]
  ge::GeTensorDesc output_desc2 = UnsortedSegmentSumdNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeShape output_desc_shape = output_desc2.GetShape();
  //output [xxx,8]
  vector<int64_t> output_desc_shape3_Dim = UnsortedSegmentSumdNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  int64_t output_desc_shape3_DimSize = output_desc_shape3_Dim.size();
  output_desc_shape.SetDim(output_desc_shape3_DimSize-1, padDimSize);
  ge::Format output_desc2_Format = output_desc2.GetFormat();
  ge::DataType output_desc2_type = output_desc2.GetDataType();
  ge::GeTensorDesc tensorDescUnsortedOutput(GeShape(), output_desc2_Format, input_desc_type);
  tensorDescUnsortedOutput.SetShape(output_desc_shape);
  tensorDescUnsortedOutput.SetOriginFormat(output_desc2_Format);
  tensorDescUnsortedOutput.SetOriginDataType(output_desc2_type);

  FUSION_PASS_CHECK(UnsortedSegmentSumd8_desc->AddOutputDesc(tensorDescUnsortedOutput) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);


  //add UnsortedSegmentSumd8_node node
  ge::NodePtr UnsortedSegmentSumd8_node = graph.AddNode(UnsortedSegmentSumd8_desc);
  fusionNodes.push_back(UnsortedSegmentSumd8_node);

  ge::AttrUtils::SetInt(UnsortedSegmentSumd8_node->GetOpDesc(), "num_segments", output_desc_shape3_Dim[0]);

  //3 copy Opdesc SlicefusedNode_desc
  std::shared_ptr<ge::OpDesc> SlicefusedNode_desc = nullptr;
  SlicefusedNode_desc =
      std::make_shared<ge::OpDesc>(UnsortedSegmentSumdNode->GetName() + "/" + "SliceLast", "SliceD");
  FUSION_PASS_CHECK(SlicefusedNode_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "SlicefusedNode_desc is null, fusion failed."),
           return PARAM_INVALID);

  // add input
  FUSION_PASS_CHECK(SlicefusedNode_desc->AddInputDesc(tensorDescUnsortedOutput) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add input failed."), return FAILED);

  // add output
  ge::GeTensorDesc output_desc3 = UnsortedSegmentSumdNode->GetOpDesc()->GetOutputDesc(0);
  //ge::GeShape output_desc_shape3= output_desc3.GetShape();
  FUSION_PASS_CHECK(SlicefusedNode_desc->AddOutputDesc(output_desc3) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "add output failed."), return FAILED);

  // add SlicefusedNode_node node
  ge::NodePtr SlicefusedNode_node = graph.AddNode(SlicefusedNode_desc);
  fusionNodes.push_back(SlicefusedNode_node);
  int64_t SliceDimsSize = output_desc_shape3_Dim.size();
  vector<int64_t> SlicedOffsets(SliceDimsSize,0);
  ge::AttrUtils::SetListInt(SlicefusedNode_node->GetOpDesc(), "offsets", SlicedOffsets);
  ge::AttrUtils::SetListInt(SlicefusedNode_node->GetOpDesc(), "size", output_desc_shape3_Dim);

  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               UnsortedSegmentSumdNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
               AicpuPad1to8fused_node->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   UnsortedSegmentSumdNode->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   AicpuPad1to8fused_node->GetName().c_str()),
           return FAILED);
  // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               AicpuPad1to8fused_node->GetOutDataAnchor(0),
               UnsortedSegmentSumd8_node->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   AicpuPad1to8fused_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   UnsortedSegmentSumd8_node->GetName().c_str()),
           return FAILED);
    // connect input edge
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               UnsortedSegmentSumdNode->GetInDataAnchor(1)->GetPeerOutAnchor(),
               UnsortedSegmentSumd8_node->GetInDataAnchor(1)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   AicpuPad1to8fused_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   AicpuPad1to8fused_node->GetName().c_str()),
           return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
               UnsortedSegmentSumd8_node->GetOutDataAnchor(0),
               SlicefusedNode_node->GetInDataAnchor(0)) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   UnsortedSegmentSumd8_node->GetInDataAnchor(0)
                       ->GetPeerOutAnchor()
                       ->GetOwnerNode()
                       ->GetName()
                       .c_str(),
                   SlicefusedNode_node->GetName().c_str()),
           return FAILED);

  // connect output edge
  for (auto inDataAnchor :
       UnsortedSegmentSumdNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(UnsortedSegmentSumdNode->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(SlicefusedNode_node->GetOutDataAnchor(0),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

// set node type
  AicpuPad1to8fused_node->GetOpDesc()->SetType("AscendPadding");
  UnsortedSegmentSumd8_node->GetOpDesc()->SetType("UnsortedSegmentSumD");
  SlicefusedNode_node->GetOpDesc()->SetType("SliceD");

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(UnsortedSegmentSumdNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove UnsortedSegmentSumdNode failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "UnsortedSegmentSumdFusionPass graph fusion success!");
  return SUCCESS;

}

REGISTER_PASS("UnsortedSegmentSumdFusionPass", BUILT_IN_GRAPH_PASS,
              UnsortedSegmentSumdFusionPass);

}
