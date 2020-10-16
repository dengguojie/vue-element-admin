#include "prior_box_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

#include "common/debug/log.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_PRIORBOX = "PriorBox";
static const char *PRIORBOX = "PriorBox";

Status GenerateWIndexFP16(const int32_t w, uint16_t *output1) {
  for (int32_t j = 0; j < w; ++j) {
    fp16_t t;
    t.val = 0;
    t = j;
    output1[j] = t.val;
  }
  return SUCCESS;
}
Status GenerateHIndexFP16(const int32_t h, uint16_t *output1) {
  for (int32_t i = 0; i < h; ++i) {
    fp16_t t;
    t.val = 0;
    t = i;
    output1[i] = t.val;
  }
  return SUCCESS;
}
Status GenerateWIndexFP32(const int32_t w, float *output1) {
  for (int32_t j = 0; j < w; ++j) {
    output1[j] = (float) j;
  }
  return SUCCESS;
}
Status GenerateHIndexFP32(const int32_t h, float *output1) {
  for (int32_t i = 0; i < h; ++i) {
    output1[i] = (float) i;
  }
  return SUCCESS;
}

Status GenerateBoxFP32(float* box_w_all, float* box_h_all,
                       vector<float>& aspectratios_new, ge::NodePtr fusedNode) {
  vector<float> min_size;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "min_size", min_size);
  int64_t min_size_size = min_size.size();
  vector<float> max_size;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "max_size", max_size);

  int count1 = 0;
  int count2 = 0;
  for(int k = 0; k < min_size_size; k++){
    float box_w_min = min_size[k];
    float box_h_min = min_size[k];
    box_w_all[count1++] = box_w_min;
    box_h_all[count2++] = box_h_min;
    if (max_size.size() > 0) {
      float box_w_max = sqrt(min_size[k] * max_size[k]);
      float box_h_max = box_w_max;
      box_w_all[count1++] = box_w_max;
      box_h_all[count2++] = box_h_max;
    }

    for(uint16_t index = 0; index < aspectratios_new.size(); index++){
      float ar = aspectratios_new[index];
      float box_w = min_size[k] * sqrt(ar);
      float box_h = min_size[k] / sqrt(ar);
      box_w_all[count1++] = box_w;
      box_h_all[count2++] = box_h;
    }
  }
        return SUCCESS;
}
Status GenerateBoxFP16(uint16_t *box_w_all, uint16_t *box_h_all,
                       vector<float>& aspectratios_new, ge::NodePtr fusedNode) {
  vector<float> min_size;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "min_size", min_size);
  int64_t min_size_size = min_size.size();
  vector<float> max_size;
  ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "max_size", max_size);

  int count1 = 0;
  int count2 = 0;
  for(int k = 0; k < min_size_size; k++){
    fp16_t t_box_w;
    fp16_t t_box_h;
    t_box_w.val = 0;
    t_box_h.val = 0;

    float box_w_min = min_size[k];
    float box_h_min = min_size[k];
    t_box_w = box_w_min;
    t_box_h = box_h_min;
    box_w_all[count1++] = t_box_w.val;
    box_h_all[count2++] = t_box_h.val;

    if (max_size.size() > 0) {
      float box_w_max = sqrt(min_size[k] * max_size[k]);
      float box_h_max = box_w_max;
      t_box_w = box_w_max;
      t_box_h = box_h_max;
      box_w_all[count1++] = t_box_w.val;
      box_h_all[count2++] = t_box_h.val;
    }

    for(uint16_t index = 0; index < aspectratios_new.size(); index++){
      float ar = aspectratios_new[index];
      float box_w = min_size[k] * sqrt(ar);
      float box_h = min_size[k] / sqrt(ar);
      t_box_w = box_w;
      t_box_h = box_h;
      box_w_all[count1++] = t_box_w.val;
      box_h_all[count2++] = t_box_h.val;
    }
  }
        return SUCCESS;
}


vector<FusionPattern *> PriorBoxPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
    new (std::nothrow) FusionPattern("PriorBoxFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_PRIORBOX, {PRIORBOX}).SetOutput(PATTERN_PRIORBOX);

  patterns.push_back(pattern);

  return patterns;
}
Status PriorBoxPass::Fusion(ge::ComputeGraph &graph,
                                  Mapping &mapping,
                                  vector<ge::NodePtr> &newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into PriorBoxPass");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_PRIORBOX, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."), return PARAM_INVALID);

  // input of priorbox
  ge::OpDescPtr priorboxDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(priorboxDesc == nullptr,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
    return PARAM_INVALID);

  // 获取进入节点的输入描述：区分const与variable
  ge::GeTensorDesc priorboxInputTensor =
    fusedNode->GetOpDesc()->GetInputDesc(0);

  // 获取shape信息
  ge::GeShape diagInputShape = priorboxInputTensor.GetShape();

  // GESHAPE->vector
  vector<int64_t> dimInfo = diagInputShape.GetDims();
  if(dimInfo.size() == 4){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxPass feature dimInfo:%d,%d,%d,%d", dimInfo[0],
        dimInfo[1], dimInfo[2], dimInfo[3]);
  } else if(dimInfo.size() == 5){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxPass feature dimInfo:%d,%d,%d,%d,%d", dimInfo[0],
        dimInfo[1], dimInfo[2], dimInfo[3], dimInfo[4]);
  } else{
    OP_LOGE(FUSED_OP_TYPE.c_str(), "PriorBoxPass feature dim size must be 4 or 5!");
    return FAILED;
  }

    vector<float> aspect_ratio;
    ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "aspect_ratio", aspect_ratio);
    int64_t ar_size = aspect_ratio.size();
    vector<float> min_size;
    ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "min_size", min_size);
    int64_t min_size_size = min_size.size();
    vector<float> max_size;
    ge::AttrUtils::GetListFloat(fusedNode->GetOpDesc(), "max_size", max_size);
    int64_t max_size_size = max_size.size();
    bool flip;
    ge::AttrUtils::GetBool(fusedNode->GetOpDesc(), "flip", flip);

    GE_CHECK_POSITIVE_SIZE_RANGE(aspect_ratio.size());
    GE_CHECK_POSITIVE_SIZE_RANGE(min_size.size());

    vector<float> aspectratios_new;
    for(int i = 0; i < ar_size; i++){
        float ar = aspect_ratio[i];
        bool already_exist = false;
        if(fabsf(ar - 1.0) < 1e-6){
            already_exist = true;
        } else{
            for(uint16_t j = 0; j < aspectratios_new.size(); j++){
               if(fabsf(ar - aspectratios_new[j]) < 1e-6){
                   already_exist = true;
                   break;
                }
              }
        }
        if(!already_exist){
            aspectratios_new.push_back(ar);
            if(flip){
              aspectratios_new.push_back(1.0/ar);
            }
        }
    }
    int64_t ar_new_size = aspectratios_new.size();

    int64_t priorNum;
    if (ar_size == 1 && (fabsf(aspect_ratio[0] - 1.0) < 1e-6)) {
     priorNum = min_size_size * ar_size + max_size_size;
    } else{
      priorNum = min_size_size + min_size_size * ar_new_size + max_size_size;
    }

  ge::GeTensorPtr assitPtrW = nullptr;
  ge::GeTensorPtr assitPtrH = nullptr;
  ge::GeTensorPtr assitPtrBoxW = nullptr;
  ge::GeTensorPtr assitPtrBoxH = nullptr;

    unique_ptr<uint16_t[]> inputAssitW(new (std::nothrow)
      uint16_t[dimInfo[3]]());
    FUSION_PASS_CHECK(inputAssitW.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssitW is NULL"),
      return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssitH(new (std::nothrow)
      uint16_t[dimInfo[2]]());
    FUSION_PASS_CHECK(inputAssitH.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssitH is NULL"),
      return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssitBoxW(new (std::nothrow)
      uint16_t[priorNum]());
    FUSION_PASS_CHECK(inputAssitBoxW.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssitBoxW is NULL"),
      return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssitBoxH(new (std::nothrow)
      uint16_t[priorNum]());
    FUSION_PASS_CHECK(inputAssitBoxH.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssitBoxH is NULL"),
      return PARAM_INVALID);

    Status ret = GenerateWIndexFP16(dimInfo[3], inputAssitW.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "GenerateWIndex failed."), return ret);
    ret = GenerateHIndexFP16(dimInfo[2], inputAssitH.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "GenerateHIndex failed."), return ret);

    ret = GenerateBoxFP16(inputAssitBoxW.get(), inputAssitBoxH.get(), aspectratios_new, fusedNode);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "GenerateBoxFP16 failed."), return ret);

    // 定义辅助矩阵输入shape
    vector<int64_t> assitDataWDimInfo;
    assitDataWDimInfo.push_back(dimInfo[3]);
    assitDataWDimInfo.push_back(1);
    assitDataWDimInfo.push_back(1);
    assitDataWDimInfo.push_back(1);
    vector<int64_t> assitDataHDimInfo;
    assitDataHDimInfo.push_back(dimInfo[2]);
    assitDataHDimInfo.push_back(1);
    assitDataHDimInfo.push_back(1);
    assitDataHDimInfo.push_back(1);

    vector<int64_t> assitBoxDimInfo;
    assitBoxDimInfo.push_back(priorNum);
    assitBoxDimInfo.push_back(1);
    assitBoxDimInfo.push_back(1);
    assitBoxDimInfo.push_back(1);

    ge::GeShape assitDataWShape(assitDataWDimInfo);
    ge::GeShape assitDataHShape(assitDataHDimInfo);
    ge::GeShape assitBoxShape(assitBoxDimInfo);
    ge::GeTensorDesc tensorDescW(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensorDescW.SetShape(assitDataWShape);
    tensorDescW.SetOriginFormat(priorboxInputTensor.GetFormat());
    tensorDescW.SetOriginDataType(DT_FLOAT16);
    ge::GeTensorDesc tensorDescH(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensorDescH.SetShape(assitDataHShape);
    tensorDescH.SetOriginFormat(priorboxInputTensor.GetFormat());
    tensorDescH.SetOriginDataType(DT_FLOAT16);
    ge::GeTensorDesc tensorDescBox(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensorDescBox.SetShape(assitBoxShape);
    tensorDescBox.SetOriginFormat(priorboxInputTensor.GetFormat());
    tensorDescBox.SetOriginDataType(DT_FLOAT16);

    FUSION_PASS_MAKE_SHARED((assitPtrW =
      std::make_shared<ge::GeTensor>(tensorDescW,
      reinterpret_cast<uint8_t *>(inputAssitW.get()),
      dimInfo[3] * sizeof(uint16_t))),
      assitPtrW = nullptr; return PARAM_INVALID);
    FUSION_PASS_MAKE_SHARED((assitPtrH =
      std::make_shared<ge::GeTensor>(tensorDescH,
      reinterpret_cast<uint8_t *>(inputAssitH.get()),
      dimInfo[2] * sizeof(uint16_t))),
      assitPtrH = nullptr; return PARAM_INVALID);

    FUSION_PASS_MAKE_SHARED((assitPtrBoxW =
      std::make_shared<ge::GeTensor>(tensorDescBox,
      reinterpret_cast<uint8_t *>(inputAssitBoxW.get()),
      priorNum * sizeof(uint16_t))),
      assitPtrBoxW = nullptr; return PARAM_INVALID);
    FUSION_PASS_MAKE_SHARED((assitPtrBoxH =
      std::make_shared<ge::GeTensor>(tensorDescBox,
      reinterpret_cast<uint8_t *>(inputAssitBoxH.get()),
      priorNum * sizeof(uint16_t))),
      assitPtrBoxH = nullptr; return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {assitPtrH, assitPtrW,
    assitPtrBoxH, assitPtrBoxW};
  ge::OpDescUtils::SetWeights(fusedNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(fusedNode);
  NodePtr constInput0 = constInputNodes[0];
  constInput0->GetOpDesc()->SetType("Const");
  NodePtr constInput1 = constInputNodes[1];
  constInput1->GetOpDesc()->SetType("Const");
  NodePtr constInput2 = constInputNodes[2];
  constInput2->GetOpDesc()->SetType("Const");
  NodePtr constInput3 = constInputNodes[3];
  constInput3->GetOpDesc()->SetType("Const");

  priorboxDesc->SetType("PriorBoxD");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "PriorBoxFusionPass pass handle success!!!!");

  return SUCCESS;
  }
REGISTER_PASS("PriorBoxPass", BUILT_IN_GRAPH_PASS,
  PriorBoxPass);
}