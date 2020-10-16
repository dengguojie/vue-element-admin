
#include <vector>
#include <nlohmann/json.hpp>
#include <math.h>
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/attr_utils.h"
#include "tbe_aipp_fusion_rule.h"

namespace fe {
  /***************************************************************
  check! strideh optim case, aipp can not fusion with conv.
  ***************************************************************/
bool TbeAippFusionRule::CheckAippConvStridehValidation(ge::NodePtr convNode) {
  ge::Format firstFormat = convNode->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  vector<int64_t> firstDims(4);
  firstDims = convNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> pads(4);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "pads", pads),
           OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s pads attr not success.",
                     convNode->GetName().c_str()), return false);
  vector<int64_t> strides(4);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "strides",
                                        strides),
           OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s strides attr not success.",
                     convNode->GetName().c_str()), return false);
  int64_t padUp = pads[0];
  int64_t padDown = pads[1];
  int64_t padLeft = pads[2];
  int64_t padRight = pads[3];

  int64_t strideH = 0;
  if (firstFormat == ge::FORMAT_NCHW){
    strideH = strides[2];
  } else if (firstFormat == ge::FORMAT_NHWC){
    strideH = strides[1];
  }
  ge::Format secondFormat = convNode->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> secondDims(4);
  secondDims = convNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  int64_t heightFt = 0;
  if (secondFormat == ge::FORMAT_NCHW){
    heightFt = secondDims[2];
  } else if (secondFormat == ge::FORMAT_NHWC){
    heightFt = secondDims[1];
  } else if (secondFormat == ge::FORMAT_HWCN){
    heightFt = secondDims[0];
  }
  ge::Format filterFormat = convNode->GetOpDesc()->GetInputDesc(1).GetFormat();
  int64_t padSum = padUp + padDown + padLeft + padRight;
  bool strideh_opti_flag = (heightFt == 1 && strideH > 1 && padSum == 0 && filterFormat != ge::FORMAT_FRACTAL_Z_C04);
  FUSION_PASS_CHECK(strideh_opti_flag == true,
           OP_LOGI(convNode->GetType().c_str(), "node[%s]'s is the strideh optim case"
                   "can not fusion.",
                   convNode->GetName().c_str()),
           return false);
  return true;
}
/***************************************************************
check! load2d case, aipp can not fusion with conv.
***************************************************************/
bool TbeAippFusionRule::CheckConvload2dNodeValidation(ge::NodePtr convNode) {
  ge::Format firstFormat = convNode->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  vector<int64_t> strides(4);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "strides",
                                        strides),
             OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s strides attr not success.",
                     convNode->GetName().c_str()), return false);
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (firstFormat == ge::FORMAT_NCHW){
    strideH = strides[2];
    strideW = strides[3];
  } else if (firstFormat == ge::FORMAT_NHWC){
    strideH = strides[1];
    strideW = strides[2];
  }
  bool strideFlg = (strideH == 1 && strideW == 1);

  ge::Format secondFormat =
          convNode->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> secondDims =
          convNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  bool filterFlg = false;
  if (secondFormat == ge::FORMAT_NCHW) {
    if(secondDims[2] == 1 && secondDims[3] == 1) {
       filterFlg = true;
    }

  } else if (secondFormat == ge::FORMAT_NHWC) {
     if(secondDims[1] == 1 && secondDims[2] == 1) {
       filterFlg = true;
    }
  } else {
     if(secondDims[0] == 1 && secondDims[1] == 1) {
       filterFlg = true;
    }
}

vector<int64_t> pads(4);
FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "pads",
                                        pads),
         OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s pads attr not success.",
                     convNode->GetName().c_str()), return false);
int64_t padUp = pads[0];
int64_t padDown = pads[1];
int64_t padLeft = pads[2];
int64_t padRight = pads[3];
bool padFlg = (padUp == 0 && padDown == 0 && padLeft == 0 && padRight == 0);
ge::DataType secondDataType = convNode->GetOpDesc()->GetInputDesc(0).GetDataType();
bool load2d_flg = filterFlg && strideFlg && padFlg && secondDataType == ge::DT_FLOAT16;
FUSION_PASS_CHECK(load2d_flg == true,
         OP_LOGI(convNode->GetType().c_str(), "node[%s]'s is the load2d case"
                   "can not fusion.",
                   convNode->GetName().c_str()),
           return false);
  return true;
}
/***************************************************************
if the minimal l1 buffer is exceed the L1 Buffer Size,
the aipp can not fusion with the conv.
***************************************************************/
bool TbeAippFusionRule::CheckAippConvEltwiseFusionValidation(ge::NodePtr convNode,
  const string &inputFormat){
  ge::Format firstFormat = convNode->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  vector<int64_t> firstDims(4);
  firstDims = convNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> dilations(4);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "dilations",
                                        dilations),
             OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s dilations attr not success.",
                     convNode->GetName().c_str()), return false);
  vector<int64_t> strides(4);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "strides",
                                        strides),
           OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s strides attr not success.",
                     convNode->GetName().c_str()), return false);
  int64_t widthFm = 0;
  int64_t dilateH = 0;
  int64_t dilateW = 0;
  int64_t strideH = 0;
  int64_t strideW = 1;
  if (firstFormat == ge::FORMAT_NCHW){
    widthFm = firstDims[3];
    dilateH = dilations[2];
    dilateW = dilations[3];
    strideH = strides[2];
    strideW = strides[3];
  } else if (firstFormat == ge::FORMAT_NHWC){
    widthFm = firstDims[2];
    dilateH = dilations[1];
    dilateW = dilations[2];
    strideH = strides[1];
    strideW = strides[2];
  }
  ge::Format secondFormat = convNode->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> secondDims(4);
  secondDims = convNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  int64_t heightFt = 0;
  int64_t widthFt = 0;
  if (secondFormat == ge::FORMAT_NCHW){
    heightFt = secondDims[2];
    widthFt = secondDims[3];
  } else if (secondFormat == ge::FORMAT_NHWC){
    heightFt = secondDims[1];
    widthFt = secondDims[2];
  } else if (secondFormat == ge::FORMAT_HWCN){
    heightFt = secondDims[0];
    widthFt = secondDims[1];
  }
  vector<int64_t> pads(4);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), "pads",
                                        pads),
           OP_LOGI(convNode->GetType().c_str(), "Get node[%s]'s pads attr not success.",
                     convNode->GetName().c_str()), return false);
  int64_t padLeft = pads[2];
  int64_t padRight = pads[3];
  int64_t wkDilation = (widthFt - 1)*dilateW + 1;
  int64_t hkDilation = (heightFt - 1)*dilateH + 1;
  int64_t widthOut = floor((widthFm - wkDilation + padLeft + \
        padRight) / strideW) + 1;
  int64_t widthIn = floor(16 / widthOut) + 2;
  int64_t tmp = ((widthIn - 1)*strideH + hkDilation)*widthFm;
  if (inputFormat == "YUV420SP_U8"){
    tmp = tmp + 2*widthFm;
  }
  int64_t mBitRatio = 2;
  int64_t ci0 = 16;
  int64_t maxFeatureMapL1 = ci0*tmp*mBitRatio;
  int64_t maxL1 = 0;
  PlatformInfo platformInfo;
  OptionalInfo optiCompilationInfo;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo,
                                      optiCompilationInfo) != fe::SUCCESS ,
            OP_LOGI(convNode->GetType().c_str(), "Get platformInfo failed."),
            return fe::FAILED);
  maxL1 = platformInfo.aiCoreSpec.l1Size;
  FUSION_PASS_CHECK(maxFeatureMapL1 > maxL1,
           OP_LOGI(convNode->GetType().c_str(), "node[%s]'s minimal l1 buffer is exceed the L1 Buffer Size"
                   "can not fusion.",
                   convNode->GetName().c_str()),
           return false);
  return true;
}

}