#include "matrix_set_diag_fusion_pass.h"

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
#include "fp16_t.hpp"
#include "pattern_fusion_util.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
  // TF的算子名
  static const float FLOAT_NUM_ZERO = 0;
  static const int32_t INT_NUM_ZERO = 0;
  static const uint16_t UINT_NUM_ZERO = 0;
  static const int8_t INT8_NUM_ZERO = 0;
  static const uint8_t UINT8_NUM_ZERO = 0;
  static const string PATTERN_MATRIXSETDIAG = "MatrixSetDiag";
  static const std::string CONSTANTOP = "Constant";
  static const char *MATRIXSETDIAG = "MatrixSetDiag";

  template<typename Dtype>
  Status AssitHelp(const int32_t n, const int32_t m, const int32_t x, Dtype& output1) {
    Dtype* output = &output1;
    int32_t z = m * x;
    int32_t y = n / z;
    if (m > x){
      for (int32_t i = 0; i < y; ++i) {
        for (int32_t j = 0; j < x; ++j) {
          output[(1+m)*j+m*x*i] = 1;
        }
      }
    }

    else {
      for (int32_t i = 0; i < y; ++i) {
        for (int32_t j = 0; j < m; ++j) {
          output[(1+m)*j+m*x*i] = 1;
        }
      }
    }

    return SUCCESS;
  }

  template<typename Dtype>
  Status AssitHelp1(const int32_t n, const int32_t m, const int32_t x, Dtype& output1) {
    Dtype* output = &output1;
    fp16_t t;
    t.val = 0;
    int32_t xx = 1;
    t = xx;
    int32_t z = m * x;
    int32_t y = n / z;
    if (m > x){
      for (int32_t i = 0; i < y; ++i) {
        for (int32_t j = 0; j < x; ++j) {
          output[(1+m)*j+m*x*i] = t.val;
        }
      }
    }

    else {
      for (int32_t i = 0; i < y; ++i) {
        for (int32_t j = 0; j < m; ++j) {
          output[(1+m)*j+m*x*i] = t.val;
        }
      }
    }

    return SUCCESS;
  }

  vector<FusionPattern*> MatrixSetDiagFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;

    // matrix_set_diag->matrix_set_diag_d
    // define MatrixSetDiagFusion
    FusionPattern *pattern = new (std::nothrow)
                             FusionPattern("MatrixSetDiagFusionPass");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
             return patterns);

    // define origin graph
    pattern->AddOpDesc(PATTERN_MATRIXSETDIAG, {MATRIXSETDIAG})
      .SetOutput(PATTERN_MATRIXSETDIAG);

    patterns.push_back(pattern);

    return patterns;
  }

Status MatrixSetDiagFusionPass::Fusion(ge::ComputeGraph &graph,
                                       Mapping &mapping,
                                       vector<ge::NodePtr> &fusionNodes)
{
    // matrix_set_diag node
    ge::NodePtr matrixsetdiagVNode = GetNodeFromMapping(PATTERN_MATRIXSETDIAG, mapping);
    FUSION_PASS_CHECK(matrixsetdiagVNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "matrixsetdiagVNode is null, " \
             "fusion failed."), return PARAM_INVALID);

    // input of matrix_set_diag
    ge::OpDescPtr matrixsetdiagDesc = matrixsetdiagVNode->GetOpDesc();
    FUSION_PASS_CHECK(matrixsetdiagDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "matrixsetdiagVNode's OpDesc is " \
             "null, fusion failed."), return PARAM_INVALID);
    ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(matrixsetdiagDesc);
    // 寻找matrix_set_diag节点的输入的父节点
    ge::InDataAnchorPtr matrixsetdiagAnchorPtr0 = matrixsetdiagVNode->GetInDataAnchor(0);
    ge::OutDataAnchorPtr constAnchorPtr0 =
      matrixsetdiagAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr constNode0 = constAnchorPtr0->GetOwnerNode();

    // 获取进入matrix_set_diag节点的输入描述：区分const与variable
    ge::GeTensorDesc matrixsetdiagInputTensor =
      constNode0->GetOpDesc()->GetOutputDesc(0);

    // 获取shape信息
    ge::GeShape matrixsetdiagInputShape = matrixsetdiagInputTensor.GetShape();

    // 获取数据类型
    DataType dataType = matrixsetdiagInputTensor.GetDataType();

    // 获取输入shape的维度累乘
    int64_t dimNums = 1;
    int64_t dimsInput = matrixsetdiagInputShape.GetDimNum()-1;
    int64_t dimsInput1 = matrixsetdiagInputShape.GetDimNum()-2;
    for (size_t j = 0; j < matrixsetdiagInputShape.GetDimNum(); ++j) {
      dimNums = matrixsetdiagInputShape.GetDim(j) * dimNums;
    }


    // 获取输入shape的最后两个维度
    int64_t dimNums1 = matrixsetdiagInputShape.GetDim(dimsInput);
    int64_t dimNums2 = matrixsetdiagInputShape.GetDim(dimsInput1);

    // GESHAPE->vector
    vector<int64_t> dimInfo = matrixsetdiagInputShape.GetDims();

    // Format
    Format  assitMatrixFormat = matrixsetdiagInputTensor.GetFormat();

    ge::GeTensorPtr assitPtr = nullptr;
    // 应采用泛型编程
    if (dataType == ge::DT_FLOAT) {
      unique_ptr<float[]> inputAssit(new (std::nothrow)
        float[dimNums]());
      FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
               return PARAM_INVALID);
      Status ret = NnSet(dimNums, FLOAT_NUM_ZERO,
                         *reinterpret_cast<float*>(inputAssit.get()));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

      ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

      // 定义辅助矩阵输入shape
      ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
      ge::GeShape assitShape = matrixsetdiagInputShape;
      tensorDesc.SetShape(assitShape);
      tensorDesc.SetOriginShape(assitShape);
      tensorDesc.SetFormat(assitMatrixFormat);
      tensorDesc.SetOriginFormat(assitMatrixFormat);
      tensorDesc.SetDataType(ge::DT_FLOAT);
      tensorDesc.SetOriginDataType(ge::DT_FLOAT);
      FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                     reinterpret_cast<uint8_t *>(inputAssit.get()),
                    dimNums * sizeof(float))), assitPtr = nullptr;
                    return PARAM_INVALID);
    } else if (dataType == ge::DT_INT32) {
      unique_ptr<int32_t[]> inputAssit(new (std::nothrow)
                                       int32_t[dimNums]());
      FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
               return PARAM_INVALID);

      Status ret = NnSet(dimNums, INT_NUM_ZERO,
                         *reinterpret_cast<int32_t*>(inputAssit.get()));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

      ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

      // 定义辅助矩阵输入shape
      ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_INT32);
      ge::GeShape assitShape = matrixsetdiagInputShape;
      tensorDesc.SetShape(assitShape);
      tensorDesc.SetOriginShape(assitShape);
      tensorDesc.SetFormat(assitMatrixFormat);
      tensorDesc.SetOriginFormat(assitMatrixFormat);
      tensorDesc.SetDataType(ge::DT_INT32);
      tensorDesc.SetOriginDataType(ge::DT_INT32);
      FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                     reinterpret_cast<uint8_t *>(inputAssit.get()),
                     dimNums * sizeof(int32_t))),
                     assitPtr = nullptr; return PARAM_INVALID);
    } else if (dataType == ge::DT_FLOAT16) {
      unique_ptr<uint16_t[]> inputAssit(new (std::nothrow)
                                        uint16_t[dimNums]());
      FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
               return PARAM_INVALID);
      Status ret = NnSet(dimNums, UINT_NUM_ZERO,
                         *reinterpret_cast<uint16_t*>(inputAssit.get()));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

      ret = AssitHelp1(dimNums, dimNums1, dimNums2, *inputAssit.get());
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

      // 定义辅助矩阵输入shape
      ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
      ge::GeShape assitShape = matrixsetdiagInputShape;
      tensorDesc.SetShape(assitShape);
      tensorDesc.SetOriginShape(assitShape);
      tensorDesc.SetFormat(assitMatrixFormat);
      tensorDesc.SetOriginFormat(assitMatrixFormat);
      tensorDesc.SetDataType(ge::DT_FLOAT16);
      tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
      FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                     reinterpret_cast<uint8_t *>(inputAssit.get()),
                     dimNums * sizeof(uint16_t))),
                     assitPtr = nullptr; return PARAM_INVALID);
    } else if (dataType == ge::DT_INT8) {
      unique_ptr<int8_t[]> inputAssit(new (std::nothrow)
                                       int8_t[dimNums]());
      FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
               return PARAM_INVALID);

      Status ret = NnSet(dimNums, INT8_NUM_ZERO,
                         *reinterpret_cast<int8_t*>(inputAssit.get()));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

      ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

      // 定义辅助矩阵输入shape
      ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_INT8);
      ge::GeShape assitShape = matrixsetdiagInputShape;
      tensorDesc.SetShape(assitShape);
      tensorDesc.SetOriginShape(assitShape);
      tensorDesc.SetFormat(assitMatrixFormat);
      tensorDesc.SetOriginFormat(assitMatrixFormat);
      tensorDesc.SetDataType(ge::DT_INT8);
      tensorDesc.SetOriginDataType(ge::DT_INT8);
      FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                     reinterpret_cast<uint8_t *>(inputAssit.get()),
                     dimNums * sizeof(int8_t))),
                     assitPtr = nullptr; return PARAM_INVALID);
    } else if (dataType == ge::DT_UINT8) {
      unique_ptr<uint8_t[]> inputAssit(new (std::nothrow)
                                       uint8_t[dimNums]());
      FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
               return PARAM_INVALID);

      Status ret = NnSet(dimNums, UINT8_NUM_ZERO,
                         *reinterpret_cast<uint8_t*>(inputAssit.get()));
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

      ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

      // 定义辅助矩阵输入shape
      ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_UINT8);
      ge::GeShape assitShape = matrixsetdiagInputShape;
      tensorDesc.SetShape(assitShape);
      tensorDesc.SetOriginShape(assitShape);
      tensorDesc.SetFormat(assitMatrixFormat);
      tensorDesc.SetOriginFormat(assitMatrixFormat);
      tensorDesc.SetDataType(ge::DT_UINT8);
      tensorDesc.SetOriginDataType(ge::DT_UINT8);
      FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(tensorDesc,
                     reinterpret_cast<uint8_t *>(inputAssit.get()),
                     dimNums * sizeof(uint8_t))),
                     assitPtr = nullptr; return PARAM_INVALID);
    }
    // check op support
    vector< ge::GeTensorPtr > weights = { assitPtr };
    ge::OpDescUtils::SetWeights(matrixsetdiagVNode, weights);
    auto constInputNodes = OpDescUtils::GetConstInputs(matrixsetdiagVNode);
    NodePtr constInput = constInputNodes[0];
    constInput->GetOpDesc()->SetType(CONSTANTOP);
    matrixsetdiagDesc->SetType("MatrixSetDiagD");

    return SUCCESS;
  }

  REGISTER_PASS("MatrixSetDiagFusionPass", BUILT_IN_GRAPH_PASS, MatrixSetDiagFusionPass);
}  // namespace fe
