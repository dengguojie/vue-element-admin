/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief diag fusedbatchnorm_assignmoving pass
 *
 * @author z00522339
 */

#include "fusedbatchnorm_assignmoving_fusion_pass.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_registry.h"
#include "external/graph/types.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {

static const char* SUB = "Sub";
static const char* MULTIPLY = "Multiply";
static const char* ASSIGNSUB = "AssignSub";
static const char* FUSEDBATCHNORM = "FusedBatchNorm";

const std::string TENSORFLOW_ATTR_VALUE = "value";
const std::string BATCHNORM_ATTR_IS_TRAINING_FUSION = "is_training_fusion";
const std::string BATCHNORM_ATTR_MOVING_AVERAGE_FRACTION = "moving_average_fraction";

FusedBatchNormAssignMovingFusionPass::~FusedBatchNormAssignMovingFusionPass() {}

vector<FusionPattern *> FusedBatchNormAssignMovingFusionPass::DefinePatterns()
{
  vector<FusionPattern *> patterns;
  return patterns;
}

bool FusedBatchNormAssignMovingFusionPass::Init()
{
  patternAssignMean.clear();
  patternAssignVar.clear();
  AddOpDesc(PATTERN_MOVINGMEAN_SUB, {SUB}, patternAssignMean)
          .AddOpDesc(PATTERN_MOVINGMEAN_MUL, {MULTIPLY}, patternAssignMean)
          .AddOpDesc(PATTERN_MOVINGMEAN_ASSIGNSUB, {ASSIGNSUB}, patternAssignMean)
          .AddOpDesc(PATTERN_MOVINGVAR_SUB, {SUB}, patternAssignVar)
          .AddOpDesc(PATTERN_MOVINGVAR_MUL, {MULTIPLY}, patternAssignVar)
          .AddOpDesc(PATTERN_MOVINGVAR_ASSIGNSUB, {ASSIGNSUB}, patternAssignVar);

  keyOpDesc.id=PATTERN_FUSEDBATCHNORM;
  keyOpDesc.types={FUSEDBATCHNORM};
  keyOpDesc.repeatable = false;
  keyOpDesc.is_output = false;

  return (!has_error_);
}

Status FusedBatchNormAssignMovingFusionPass::Run(ge::ComputeGraph& graph)
{
  FUSION_PASS_CHECK(!Init(), OP_LOGI(FUSED_OP_TYPE.c_str(), "FusedBatchNormAssignMovingFusionPass pattern build failed."), return NOT_CHANGED);
  Mappings mappings;
  bool changed = false;
  if (!MatchAll(graph, mappings)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "FusedBatchNormAssignMovingFusion find no pattern matched.");
    return NOT_CHANGED;
  }
  // print matched result
  DumpMappings(mappings);
  // do fusion
  for (Mapping mapping : mappings) {
    vector<ge::NodePtr> fusNodes;
    Status status = Fusion(graph, mapping, fusNodes);
    if (status != SUCCESS && status != NOT_CHANGED) {
      // has error when do fusion
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion pattern FusedBatchNormAssignMovingFusionPas failed, status:%d", status);
      return status;
    }
    if(status == SUCCESS)
    {
      std::vector<ge::NodePtr> originalNodes;
      for (auto &item : mapping) {
        if (!item.second.empty()) {
          for(auto node:item.second) {
            originalNodes.push_back(node);
          }
        }
      }
      SetDataDumpAttr(originalNodes,fusNodes);

    }
    changed = changed || status == SUCCESS;
  }
  return changed ? SUCCESS : NOT_CHANGED;
}


bool FusedBatchNormAssignMovingFusionPass::MatchAll(ge::ComputeGraph& graph, Mappings &mappings)
{
  vector<ge::NodePtr> matched_key_nodes;

  for (ge::NodePtr node : graph.GetDirectNode()) {
    if (IsOpTypeExist(node->GetOpDesc()->GetType(), keyOpDesc.types)) {
      matched_key_nodes.push_back(node);
    }
  }
  // if can not find key nodes, return
  if (matched_key_nodes.empty()) {
    return false;
  }
  // match Pattern to each key node
  for (ge::NodePtr output_node : matched_key_nodes) {
    Mapping mapping;
    if (IsFusedBatchNormMatched(output_node, mapping)) {
      mappings.push_back(mapping);

      // Get output node and record map
      for (const auto &item : mapping) {
        if (item.second[0]->GetType().compare(ASSIGNSUB) == 0) {
          for (const auto nodePtr : item.second) {
            RecordOutputAnchorMap(nodePtr);
          }
        }
      }
    }
  }
  // if matched sucess, return true
  return !mappings.empty();
}

bool FusedBatchNormAssignMovingFusionPass::IsFusedBatchNormMatched(ge::NodePtr fusedBatchNormnode, Mapping &mapping)
{
  FUSION_PASS_CHECK(fusedBatchNormnode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedBatchNormnode is nullptr."), return FAILED);

  shared_ptr<OpDesc> opDesc = nullptr;
  opDesc = std::make_shared<OpDesc>(keyOpDesc);
  FUSION_PASS_CHECK(opDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "opDesc is nullptr."), return FAILED);

  mapping[opDesc].push_back(fusedBatchNormnode);

  auto outDataAnchors = fusedBatchNormnode->GetAllOutDataAnchors();

  bool isAssignMeanMatched = false;
  bool isAssignVarMatched = false;

  for (auto outAnchor: outDataAnchors) {

    if (outAnchor->GetPeerInDataAnchors().size() == 0) {
      continue;
    }
    if (outAnchor->GetIdx() == 1) {
      isAssignMeanMatched = IsAssignMovingMatched(outAnchor->GetPeerInDataAnchors().at(0)->GetOwnerNode(), patternAssignMean, mapping);
    }
    if (outAnchor->GetIdx() == 2) {
      isAssignVarMatched = IsAssignMovingMatched(outAnchor->GetPeerInDataAnchors().at(0)->GetOwnerNode(), patternAssignVar, mapping);
    }
  }
  return isAssignMeanMatched && isAssignVarMatched;
}


bool FusedBatchNormAssignMovingFusionPass::IsAssignMovingMatched(ge::NodePtr node, vector<shared_ptr<OpDesc>> pattern, Mapping &mapping)
{
  vector<ge::NodePtr > candidateNodes;
  shared_ptr<OpDesc> opDesc= nullptr;
  if (!pattern.empty() && node != nullptr) {
    opDesc = pattern.front();
    if ((IsOpTypeExist(GetNodeType(node), pattern.front()->types))) {
      mapping[opDesc].push_back(node);
      pattern.erase(pattern.begin());
      candidateNodes.push_back(node);
    }

  }

  while (!candidateNodes.empty() && !pattern.empty()) {
    // get first candidate Node
    ge::NodePtr candidateNode = candidateNodes.front();
    opDesc = pattern.front();

    FUSION_PASS_CHECK(candidateNode->GetAllOutDataAnchors().empty(), OP_LOGI(FUSED_OP_TYPE.c_str(), "candidateNode->GetAllOutDataAnchors() is empty"),
             return false);

    // sort edge by anchor_id
    auto outDataAnchorPtrList = candidateNode->GetAllOutDataAnchors();

    std::sort(outDataAnchorPtrList.begin(), outDataAnchorPtrList.end(), [](ge::OutDataAnchorPtr a, ge::OutDataAnchorPtr b)
    {
      return a->GetIdx() < b->GetIdx();
    });

    for (const auto &outAnchor : outDataAnchorPtrList) {
      for(const auto &inAnchor: outAnchor->GetPeerInDataAnchors())
      {
        ge::NodePtr nextNode = inAnchor->GetOwnerNode();
        if ((IsOpTypeExist(GetNodeType(nextNode), opDesc->types) || opDesc->types.empty())) {
          // Some Nodes may be used as input for multiple Nodes. Here, duplicate matching is avoided through IsMatched
          if (!IsMatched(opDesc, nextNode, mapping)) {
            candidateNodes.push_back(nextNode);

            // record matched node
            mapping[opDesc].push_back(nextNode);
            pattern.erase(pattern.begin());
          }
          break;
        }
      }
    }
    // remove from candidateNode if matched sucess
    candidateNodes.erase(candidateNodes.begin());
  }
  // if candidate_op_descs is empty, mean success finally
  return pattern.empty();
}


void FusedBatchNormAssignMovingFusionPass::DumpMappings(const Mappings &mappings) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Mappings of pattern FusedBatchNormAssignMovingFusionPass:");
  for (uint32_t i = 0; i < mappings.size(); i++) {
    const Mapping &mapping = mappings[i];
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Mapping: %u / %lu:", i+1, mappings.size());
    for (const auto &item : mapping) {
      auto op_desc = item.first;
      const ge::NodePtr node = item.second[0];
      if (op_desc != nullptr && node != nullptr) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "%s -> %s", op_desc->id.c_str(), node->GetName().c_str());
      }
    }
  }
}


bool FusedBatchNormAssignMovingFusionPass::IsOpTypeExist(const string &type, const vector<string> &types) {
  return find(types.begin(), types.end(), type) != types.end();
}

bool FusedBatchNormAssignMovingFusionPass::IsMatched(shared_ptr<OpDesc> op_desc, const ge::NodePtr node, const Mapping &mapping)
{
  FUSION_PASS_CHECK(op_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "op_desc is nullptr."), return false);
  const auto iter = mapping.find(op_desc);

  // firstly,op_desc exist as key in mapping, then the op_desc as key exist in vector
  return iter != mapping.end() && find(iter->second.begin(), iter->second.end(), node) != iter->second.end();
}

/**
*
* befor:
* ::tensorflow::Input x,
::tensorflow::Input scale,
::tensorflow::Input offset,
::tensorflow::Input mean,(empty, when train single op)
::tensorflow::Input variance,(empty, when train single op)
after:
 ::tensorflow::Input x,
::tensorflow::Input scale,
::tensorflow::Input offset,
::tensorflow::Input moving_mean,(as moving_mean's input(output) when train fusion op)
::tensorflow::Input moving_variance,(as moving_var's input(output) when train fusion op)
*
* @param graph
* @param mapping
* @return
*/

Status FusedBatchNormAssignMovingFusionPass::Fusion(
    ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter FusedBatchNormAssignMovingFusionPass!");
  ge::NodePtr movingMeanSub = GetNodeFromMapping(PATTERN_MOVINGMEAN_SUB, mapping);
  ge::NodePtr movingMeanMul = GetNodeFromMapping(PATTERN_MOVINGMEAN_MUL, mapping);
  ge::NodePtr movingMeanAssignSub = GetNodeFromMapping(PATTERN_MOVINGMEAN_ASSIGNSUB, mapping);

  ge::NodePtr movingVarSub = GetNodeFromMapping(PATTERN_MOVINGVAR_SUB, mapping);
  ge::NodePtr movingVarMul = GetNodeFromMapping(PATTERN_MOVINGVAR_MUL, mapping);
  ge::NodePtr movingVarAssignSub = GetNodeFromMapping(PATTERN_MOVINGVAR_ASSIGNSUB, mapping);

  ge::NodePtr fusedBatchNorm = GetNodeFromMapping(PATTERN_FUSEDBATCHNORM, mapping);
  if (domi::OpRegistry::Instance()->GetImplyType(fusedBatchNorm->GetType()) != domi::ImplyType::CCE) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion Op %s is not CCE, no need to fusion.",fusedBatchNorm->GetName().c_str());
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(movingMeanSub == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "movingMeanSub is nullptr."), return FAILED);
  FUSION_PASS_CHECK(movingMeanMul == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "movingMeanMul is nullptr."), return FAILED);
  FUSION_PASS_CHECK(movingMeanAssignSub == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "movingMeanAssignSub is nullptr."), return FAILED);
  FUSION_PASS_CHECK(movingVarSub == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "movingVarSub is nullptr."), return FAILED);
  FUSION_PASS_CHECK(movingVarMul == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "movingVarMul is nullptr."), return FAILED);
  FUSION_PASS_CHECK(movingVarAssignSub == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "movingVarAssignSub is nullptr."), return FAILED);
  FUSION_PASS_CHECK(fusedBatchNorm == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedBatchNorm is nullptr."), return FAILED);


  float exponentialAverageFactor = 0;
  FUSION_PASS_CHECK(!(movingMeanSub->GetAllInDataAnchors().size() == 2),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Input nodes of moveMean Sub node not equal to 2, exit FusedBatchNormAssignMovingFusionPass with status: not changed."),return NOT_CHANGED);
  FUSION_PASS_CHECK(!(movingMeanMul->GetInDataNodes().size() == 2),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Input nodes of moveMean Mul node not equal to 2, exit FusedBatchNormAssignMovingFusionPass with status: not changed."),return NOT_CHANGED);

  FUSION_PASS_CHECK(!(movingVarSub->GetAllInDataAnchors().size() == 2),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Input nodes of movingVar Sub node not equal to 2, exit FusedBatchNormAssignMovingFusionPass with status: not changed."),return NOT_CHANGED);

  FUSION_PASS_CHECK(!(fusedBatchNorm->GetAllInDataAnchors().size() == 5),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Input nodes of fusedBatchNorm node not equal to 5, exit FusedBatchNormAssignMovingFusionPass with status: not changed."),return NOT_CHANGED);

  Status ret;
  // remove moveMean Sub node
  for (auto inDataAnchor :movingMeanSub->GetAllInDataAnchors()) {
    if (inDataAnchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    ge::NodePtr inNode = inDataAnchor->GetPeerOutAnchor()->GetOwnerNode();
    if (inNode->GetType() != FUSEDBATCHNORM) {
      FUSION_PASS_CHECK(
          SUCCESS !=
              fusedBatchNorm->GetOpDesc()->UpdateInputDesc(3, inNode->GetOutDataAnchor(0)
                  ->GetOwnerNode()
                  ->GetOpDesc()
                  ->GetOutputDesc(0)),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "update input failed."), return FAILED);
      ret = ge::GraphUtils::RemoveEdge(fusedBatchNorm->GetInDataAnchor(3)->GetPeerOutAnchor(), fusedBatchNorm->GetInDataAnchor(3));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between mean input node and fusedBatchNorm node failed."), return ret);
      ret = ge::GraphUtils::AddEdge(inNode->GetOutDataAnchor(0), fusedBatchNorm->GetInDataAnchor(3));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between moving_mean input node and fusedBatchNorm node failed."), return ret);
    }
    ret = ge::GraphUtils::RemoveEdge(inDataAnchor->GetPeerOutAnchor(),inDataAnchor);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between %s and %s failed.", inNode->GetName().c_str(), inDataAnchor->GetOwnerNode()->GetName().c_str()), return ret);
  }
  ret = graph.RemoveNode(movingMeanSub);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove moveMean Sub node failed."), return ret);

  // remove moveMean Mul node, extract factor from Mul input node
  bool extractParaSuccess = false;
  for(auto inNode: movingMeanMul->GetInDataNodes())
  {
    if((inNode->GetType() == "Constant") || (inNode->GetType() == "Const"))
    {
      ret = ParseParaFromConst(inNode, exponentialAverageFactor, 0);
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Parse param from const failed."), return ret);
      extractParaSuccess = true;
    }

  }
  FUSION_PASS_CHECK(!extractParaSuccess, OP_LOGE(FUSED_OP_TYPE.c_str(), "extract factor failed."), return FAILED);
  ret = graph.RemoveNode(movingMeanMul);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove moveMean Mul node failed."), return ret);
  // remove moveMean AssignSub node
  ge::NodePtr meanRefNode = movingMeanAssignSub->GetOutNodes().at(0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "movingMeanAssignSub's peer outNode is %s.", meanRefNode->GetName().c_str());
  ret = graph.RemoveNode(movingMeanAssignSub);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove moveMean AssignSub node failed."), return ret);
  ret = graph.RemoveNode(meanRefNode);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove meanRefNode %s failed.",
          meanRefNode->GetName().c_str()), return ret);

  // remove moveVar Sub node
  for (auto inDataAnchor :movingVarSub->GetAllInDataAnchors()) {
    if (inDataAnchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    ge::NodePtr inNode = inDataAnchor->GetPeerOutAnchor()->GetOwnerNode();
    if (inNode->GetType() != FUSEDBATCHNORM) {
      FUSION_PASS_CHECK(
          SUCCESS !=
              fusedBatchNorm->GetOpDesc()->UpdateInputDesc(4, inNode->GetOutDataAnchor(0)
                  ->GetOwnerNode()
                  ->GetOpDesc()
                  ->GetOutputDesc(0)),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "update input failed."), return FAILED);
      ret = ge::GraphUtils::RemoveEdge(fusedBatchNorm->GetInDataAnchor(4)->GetPeerOutAnchor(), fusedBatchNorm->GetInDataAnchor(4));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between mean input node and fusedBatchNorm node failed."), return ret);
      ret = ge::GraphUtils::AddEdge(inNode->GetOutDataAnchor(0), fusedBatchNorm->GetInDataAnchor(4));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between moving_mean input node and fusedBatchNorm node failed."), return ret);
    }
    ret = ge::GraphUtils::RemoveEdge(inDataAnchor->GetPeerOutAnchor(),inDataAnchor);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "remove edge between %s and %s failed.", inNode->GetName().c_str(), inDataAnchor->GetOwnerNode()->GetName().c_str()), return ret);
  }
  ret = graph.RemoveNode(movingVarSub);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove movingVar Sub node failed."), return ret);
  // remove moveVar Mul node
  ret = graph.RemoveNode(movingVarMul);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove movingVar Mul node failed."), return ret);
  // remove moveVar AssignSub node
  ge::NodePtr varRefNode = movingVarAssignSub->GetOutNodes().at(0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "movingVarAssignSub's peer outNode is %s.", varRefNode->GetName().c_str());
  ret = graph.RemoveNode(movingVarAssignSub);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove movingVar AssignSub node failed."), return ret);
  ret = graph.RemoveNode(varRefNode);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove varRefNode %s failed.", varRefNode->GetName().c_str()),
          return ret);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "run FusedBatchNorm Fusion SUCCESS .");

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(fusedBatchNorm->GetOpDesc(), BATCHNORM_ATTR_MOVING_AVERAGE_FRACTION, exponentialAverageFactor),
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Set fusedbatchnorm exponentialAverageFactor attr failed, exit FusedBatchNormAssignMovingFusionPass with status: failed."),
          return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(fusedBatchNorm->GetOpDesc(), BATCHNORM_ATTR_IS_TRAINING_FUSION, true),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Set fusedbatchnorm is_training_fusion attr failed, exit FusedBatchNormAssignMovingFusionPass with status: failed."),
           return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "FusedBatchNormAssignMovingFusionPass success!");
  return SUCCESS;

}

FusedBatchNormAssignMovingFusionPass& FusedBatchNormAssignMovingFusionPass::AddOpDesc(const string &id, const initializer_list<string> &types, vector<shared_ptr<OpDesc>> &ops)
{
  if (id.empty()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Id cannot be empty.");
      has_error_ = true;
      return *this;
  }
  std::shared_ptr<OpDesc> op = std::make_shared<OpDesc>();
  if (op == nullptr) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "new an object failed.");
      has_error_ = true;
      return *this;
  }
  op->id = id;
  op->types = types;
  op->repeatable = false;
  op->is_output = false;
  ops.push_back(op);

  return *this;
}

std::string FusedBatchNormAssignMovingFusionPass::GetNodeType(ge::NodePtr node)
{
    return node->GetType();
}

Status FusedBatchNormAssignMovingFusionPass::ParseParaFromConst(ge::NodePtr node, float &param, int index)
{
  if (node->GetOpDesc()->GetType() == "Const" || node->GetOpDesc()->GetType() == "Constant") {
    vector<ge::ConstGeTensorPtr> weights_vec = ge::OpDescUtils::GetWeights(node);
    FUSION_PASS_CHECK(weights_vec.empty(), OP_LOGE(FUSED_OP_TYPE.c_str(), "get weights failed"), return PARAM_INVALID);
    const ge::GeTensor* tensor = weights_vec[0].get();

    /* 从tensor中获取数据 */

    int64_t dataType = 0;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(node->GetOpDesc(), "dtype", dataType), OP_LOGE(FUSED_OP_TYPE.c_str(), "Get dtype attr failed."),
             return PARAM_INVALID);
    if (dataType == (int64_t)ge::DT_FLOAT)
    {
      FUSION_PASS_CHECK(tensor->GetData().size() < (sizeof(int32_t) * (index + 1)),
               OP_LOGI(FUSED_OP_TYPE.c_str(), "data size too small."), return PARAM_INVALID);
      param = *((float*)tensor->GetData().data() + index);
    }
    else
    {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "data type not supported");
      return FAILED;
    }
    OP_LOGI(FUSED_OP_TYPE.c_str(), "FusedBatchNorm NodeDef param %f.", param);
    return SUCCESS;
  }
  return FAILED;
}

REGISTER_PASS("FusedBatchNormAssignMovingFusion", BUILT_IN_GRAPH_PASS, FusedBatchNormAssignMovingFusionPass);
}  // namespace fe
