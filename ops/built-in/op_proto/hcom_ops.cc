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
 * \file hcom_ops.cpp
 * \brief
 */
#include "inc/hcom_ops.h"

#include <string>
#include <vector>
#include <algorithm>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common_shape_fns.h"
#include "op_log.h"
#include "util/util.h"

namespace ge {
// HcomAllGather op
IMPLEMT_INFERFUNC(HcomAllGather, HcomAllGatherInferShape) {
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomAllGather", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();
  if (!ShapeFullDefined(inShape)) {
    outTensorDesc.SetShape(inShape);
    outTensorDesc.SetDataType(inTensorDesc.GetDataType());
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(opName.GetString(), "the op infershape end, shape is unknown.");
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  std::vector<int64_t> outDims;
  if (rankSize <= 0) {
    OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  outDims = inDims;
  outDims[0] = inDims[0] * rankSize;
  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomAllGather, HcomAllGatherVerify) {
  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  if (rankSize <= 0) {
    OP_LOGE(op.GetName().c_str(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomAllGather, HcomAllGatherVerify);
INFER_FUNC_REG(HcomAllGather, HcomAllGatherInferShape);

// HcomReduce op
IMPLEMT_VERIFIER(HcomReduce, HcomReduceVerify) {
  constexpr int64_t fusionAttrNoFuse = 0;
  constexpr int64_t fusionAttrFuseById = 2;
  constexpr int64_t fusionIdMinVal = -1;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;
  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "prod", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(op.GetName().c_str(), "Attr reduction [%s] is not supported. expecttd: min, max, prod, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }
  int64_t fusionAttr;
  if (op.GetAttr("fusion", fusionAttr) == GRAPH_SUCCESS) {
    if ((fusionAttr != fusionAttrNoFuse) && (fusionAttr != fusionAttrFuseById)) {
      OP_LOGE(op.GetName().c_str(), "Attr fusion [%lld] is not supported. expecttd: [%lld or %lld]", fusionAttr,
              fusionAttrNoFuse, fusionAttrFuseById);
      return GRAPH_FAILED;
    }
  }
  int64_t fusionIdAttr;
  if (op.GetAttr("fusion_id", fusionIdAttr) == GRAPH_SUCCESS) {
    if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
      OP_LOGE(op.GetName().c_str(), "Attr fusion_id [%lld] is not supported. expecttd: [%lld ~ %lld]", fusionIdAttr,
              fusionIdMinVal, fusionIdMaxVal);
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(op.GetName().c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomReduce, HcomReduceVerify);
COMMON_INFER_FUNC_REG(HcomReduce, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

// HcomAllReduce op
IMPLEMT_VERIFIER(HcomAllReduce, HcomAllReduceVerify) {
  constexpr int64_t fusionAttrMinVal = 0;
  constexpr int64_t fusionAttrMaxVal = 2;
  constexpr int64_t fusionIdMinVal = -1;
  constexpr int64_t fusionIdMaxVal = 0x7fffffff;
  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "prod", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(op.GetName().c_str(), "Attr reduction [%s] is not supported. expecttd: min, max, prod, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }
  int64_t fusionAttr;
  if (op.GetAttr("fusion", fusionAttr) == GRAPH_SUCCESS) {
    if ((fusionAttr < fusionAttrMinVal) || (fusionAttr > fusionAttrMaxVal)) {
      OP_LOGE(op.GetName().c_str(), "Attr fusion [%lld] is not supported. expecttd: [%lld ~ %lld]", fusionAttr,
              fusionAttrMinVal, fusionAttrMaxVal);
      return GRAPH_FAILED;
    }
  }
  int64_t fusionIdAttr;
  if (op.GetAttr("fusion_id", fusionIdAttr) == GRAPH_SUCCESS) {
    if ((fusionIdAttr < fusionIdMinVal) || (fusionIdAttr > fusionIdMaxVal)) {
      OP_LOGE(op.GetName().c_str(), "Attr fusion_id [%lld] is not supported. expecttd: [%lld ~ %lld]", fusionIdAttr,
              fusionIdMinVal, fusionIdMaxVal);
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(op.GetName().c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomAllReduce, HcomAllReduceVerify);
COMMON_INFER_FUNC_REG(HcomAllReduce, ELMTWISE_INFER_SHAPEANDTYPE("x", "y"));

// HcomBroadcast op
IMPLEMT_INFERFUNC(HcomBroadcast, HcomBroadcastInferShape) {
  const unsigned int UINT_MAX_VALUE = 0xFFFFFFFF;
  auto inputsSize = op.GetInputsSize();
  if (inputsSize >= UINT_MAX_VALUE) {
    OP_LOGE(op.GetName().c_str(), "GetInputsSize [%zu] is more than %u", inputsSize, UINT_MAX_VALUE);
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < inputsSize; i++) {
    auto inputDesc = op.get_dynamic_input_desc_x(i);
    auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
    auto outputDesc = opDesc->MutableOutputDesc(i);
    outputDesc->SetShape(GeShape(inputDesc.GetShape().GetDims()));
    outputDesc->SetDataType(inputDesc.GetDataType());
  }
  OP_LOGI(op.GetName().c_str(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomBroadcast, HcomBroadcastVerify) {
  OP_LOGI(op.GetName().c_str(), "the op verify end");
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(HcomBroadcast, HcomBroadcastVerify);
INFER_FUNC_REG(HcomBroadcast, HcomBroadcastInferShape);

// HcomReduceScatter op
IMPLEMT_INFERFUNC(HcomReduceScatter, HcomReduceScatterInferShape) {
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomReduceScatter", "Get op name failed.");
    return GRAPH_FAILED;
  }

  auto inTensorDesc = op.get_input_desc_x();
  auto outTensorDesc = inTensorDesc;
  auto inShape = inTensorDesc.GetShape();
  if (!ShapeFullDefined(inShape)) {
    outTensorDesc.SetShape(inShape);
    outTensorDesc.SetDataType(inTensorDesc.GetDataType());
    op.update_output_desc_y(outTensorDesc);
    OP_LOGI(opName.GetString(), "the op infershape end, shape is unknown.");
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> inDims = inShape.GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  std::vector<int64_t> outDims;
  if (rankSize <= 0) {
    OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }
  if (inDims[0] % rankSize) {
    OP_LOGE(opName.GetString(),
            "input tensor's first dim is illegal, expected: rankSize[%ld] * N "
            "(N is positive integer), actual: %ld.",
            rankSize, inDims[0]);
    return GRAPH_FAILED;
  }
  outDims = inDims;
  outDims[0] = inDims[0] / rankSize;
  ge::Shape outputShape = ge::Shape(outDims);
  ge::DataType outputDtype = inTensorDesc.GetDataType();
  outTensorDesc.SetShape(outputShape);
  outTensorDesc.SetDataType(outputDtype);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(opName.GetString(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReduceScatter, HcomReduceScatterVerify) {
  AscendString opName;
  if (op.GetName(opName) != GRAPH_SUCCESS) {
    OP_LOGE("HcomReduceScatter", "Get op name failed.");
    return GRAPH_FAILED;
  }

  std::string reduction = op.get_attr_reduction();
  const std::vector<std::string> SUPPORTED_REDUCTION = {"min", "max", "prod", "sum"};
  auto it = std::find(SUPPORTED_REDUCTION.begin(), SUPPORTED_REDUCTION.end(), reduction);
  if (it == SUPPORTED_REDUCTION.end()) {
    OP_LOGE(opName.GetString(), "Attr reduction [%s] is not supported. expected: min, max, prod, sum",
            reduction.c_str());
    return GRAPH_FAILED;
  }
  std::vector<int64_t> inDims = op.get_input_desc_x().GetShape().GetDims();
  int64_t rankSize = op.get_attr_rank_size();
  if (rankSize <= 0) {
    OP_LOGE(opName.GetString(), "attr rank_size is illegal, expected: > 0, actual: %ld.", rankSize);
    return GRAPH_FAILED;
  }
  if (inDims.size() == 0) {
    OP_LOGE(opName.GetString(), "input tensor's first dim is illegal, expected: > 0, actual: %zu.", inDims.size());
    return GRAPH_FAILED;
  }

  if (ShapeFullDefined(op.get_input_desc_x().GetShape())) {
    if (inDims[0] % rankSize) {
      OP_LOGE(opName.GetString(),
              "input tensor's first dim is illegal, expected: rankSize[%ld] * N "
              "(N is positive integer), actual:%ld.",
              rankSize, inDims[0]);
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(opName.GetString(), "the op verify end");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReduceScatter, HcomReduceScatterInferShape);
VERIFY_FUNC_REG(HcomReduceScatter, HcomReduceScatterVerify);

// HcomSend op
IMPLEMT_INFERFUNC(HcomSend, HcomSendInferShape) {
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomSend, HcomSendVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomSend, HcomSendInferShape);
VERIFY_FUNC_REG(HcomSend, HcomSendVerify);

// HcomReceive op 
IMPLEMT_INFERFUNC(HcomReceive, HcomReceiveInferShape) {
  TensorDesc outTensorDesc = op.get_output_desc_y();
  std::vector<int64_t> shapeSize{};
  op.GetAttr("shape", shapeSize);
  outTensorDesc.SetShape(ge::Shape(shapeSize));
  uint32_t dataType = op.get_attr_dtype();
  outTensorDesc.SetDataType((DataType)dataType);
  op.update_output_desc_y(outTensorDesc);
  OP_LOGI(op.GetName().c_str(), "the op infershape end");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomReceive, HcomReceiveVerify) {
  TensorDesc outTensorDesc = op.get_output_desc_y();
  std::vector<int64_t> shapeSize{};
  op.GetAttr("shape", shapeSize);
  if (shapeSize.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "Attr shape is illegal, mast be > 0");
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(HcomReceive, HcomReceiveInferShape);
VERIFY_FUNC_REG(HcomReceive, HcomReceiveVerify);

// HcomRemoteRead op
IMPLEMT_INFERFUNC(HcomRemoteRead, HcomRemoteReadInferShape) {
    std::vector<string> dep_inputs = {"remote"};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(dep_inputs);

    TensorDesc outTensorDesc = op.get_output_desc_local();
    auto inShape = op.get_input_desc_remote().GetShape();
    std::vector<int64_t> inDims = inShape.GetDims();
    std::vector<int64_t> outDims;
    outDims = inDims;
    DataType dataType = static_cast<DataType>(op.get_attr_dtype());
    int dataLength;
    switch (dataType) {
        case DT_FLOAT:
        case DT_INT32:
        case DT_UINT32:
            dataLength = 4;
            break;
        case DT_FLOAT16:
        case DT_INT16:
        case DT_UINT16:
            dataLength = 2;
            break;
        case DT_INT8:
        case DT_UINT8:
            dataLength = 1;
            break;
        case DT_DOUBLE:
        case DT_INT64:
        case DT_UINT64:
            dataLength = 8;
            break;
        default:
            dataLength = -1;
        break;
    }
    Tensor tensor;
    graphStatus state = op.GetInputConstData("remote", tensor);
    if (outDims.size() == 2) {
        if (state != GRAPH_SUCCESS) {
            outDims[1] = -1; // -1: indicate unknown
        } else {
            uint64_t* data = const_cast<uint64_t *>(reinterpret_cast<const uint64_t *>(tensor.GetData()));
            if (dataLength != -1) {
                outDims[1] = data[2] / dataLength; //  length/size_of(datatype)
            } else {
                outDims[1] = -1; // -1: indicate unknown
                std::pair<int64_t, int64_t> rangeN(10,10);
                std::pair<int64_t, int64_t> rangeX(10,40);
                std::vector<std::pair<int64_t, int64_t>> range = {rangeN, rangeX};
                outTensorDesc.SetShapeRange(range);
            }
        }
        outTensorDesc.SetShape(ge::Shape(outDims));
        outTensorDesc.SetOriginShape(ge::Shape(outDims));
    } else {
        if (!outDims.empty() && outDims[0] == -2) {
            outTensorDesc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
            outTensorDesc.SetOriginShape(ge::Shape(ge::UNKNOWN_RANK));
        } else {
            OP_LOGE(op.GetName().c_str(), "the op infershape failed");
            return GRAPH_FAILED;
        }
    }
    outTensorDesc.SetDataType(dataType);

    op.update_output_desc_local(outTensorDesc);
    OP_LOGI(op.GetName().c_str(), "the op infershape end");
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomRemoteRead, HcomRemoteReadVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomRemoteRead, HcomRemoteReadInferShape);
VERIFY_FUNC_REG(HcomRemoteRead, HcomRemoteReadVerify);

IMPLEMT_INFERFUNC(HcomRemoteRefRead, HcomRemoteRefReadInferShape) {
    std::vector<string> dep_inputs = {"remote","local_offset"};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(dep_inputs);

    TensorDesc outTensorDesc = op.get_output_desc_cache_var();
    auto inShape = op.get_input_desc_cache_var().GetShape();
    outTensorDesc.SetShape(inShape);
    outTensorDesc.SetOriginShape(inShape);
    op.update_output_desc_cache_var(outTensorDesc);
    OP_LOGI(op.GetName().c_str(), "the op infershape end");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(HcomRemoteRefRead, HcomRemoteRefReadVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomRemoteRefRead,HcomRemoteRefReadInferShape);
VERIFY_FUNC_REG(HcomRemoteRefRead, HcomRemoteRefReadVerify);

IMPLEMT_INFERFUNC(HcomRemoteScatterWrite, HcomRemoteScatterWriteInferShape) {
    std::vector<string> dep_inputs = {"remote", "local_offset"};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(dep_inputs);
    AttrUtils::SetBool(op_desc, "_force_unknown_shape", true);
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomRemoteScatterWrite, HcomRemoteScatterWriteVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomRemoteScatterWrite, HcomRemoteScatterWriteInferShape);
VERIFY_FUNC_REG(HcomRemoteScatterWrite, HcomRemoteScatterWriteVerify);

// HcomRemoteWrite op
IMPLEMT_INFERFUNC(HcomRemoteWrite, HcomRemoteWriteInferShape) {
    std::vector<string> dep_inputs = {"remote"};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(dep_inputs);
    AttrUtils::SetBool(op_desc, "_force_unknown_shape", true);
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomRemoteWrite, HcomRemoteWriteVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomRemoteWrite, HcomRemoteWriteInferShape);
VERIFY_FUNC_REG(HcomRemoteWrite, HcomRemoteWriteVerify);

// HcomAllToAllV op
IMPLEMT_INFERFUNC(HcomAllToAllV, HcomAllToAllVInferShape) {
    std::vector<string> dep_inputs = {"recv_displacements", "recv_counts"};
    auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
    opDesc->SetOpInferDepends(dep_inputs);
    AttrUtils::SetBool(opDesc, "_force_unknown_shape", true);
    Tensor recvDispTensor;
    Tensor recvCountsTensor;
    if ((op.GetInputConstData("recv_displacements", recvDispTensor) != GRAPH_SUCCESS) ||
        (op.GetInputConstData("recv_counts", recvCountsTensor) != GRAPH_SUCCESS)) {
        auto outTensorDesc = opDesc->MutableOutputDesc("recv_data");
        outTensorDesc->SetShape(GeShape(UNKNOWN_SHAPE));
        outTensorDesc->SetDataType(op.GetInputDesc("send_data").GetDataType());
        return GRAPH_SUCCESS;
    }

    DataType recvDispDtype = op.GetInputDesc("recv_displacements").GetDataType();
    vector<int64_t> recvDisp;
    GetConstValue(op, recvDispTensor, recvDispDtype, recvDisp);

    DataType recvCountsDtype = op.GetInputDesc("recv_counts").GetDataType();
    vector<int64_t> recvCounts;
    GetConstValue(op, recvCountsTensor, recvCountsDtype, recvCounts);

    std::vector<int64_t> recvDataDims;
    recvDataDims.push_back((int64_t)recvDisp.back() + (int64_t)recvCounts.back());
    auto outTensorDesc = opDesc->MutableOutputDesc("recv_data");
    outTensorDesc->SetShape(GeShape(recvDataDims));
    outTensorDesc->SetDataType(op.GetInputDesc("send_data").GetDataType());

    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomAllToAllV, HcomAllToAllVVerify) {
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomAllToAllV, HcomAllToAllVInferShape);
VERIFY_FUNC_REG(HcomAllToAllV, HcomAllToAllVVerify);

// HcomGatherAllToAllV op
IMPLEMT_INFERFUNC(HcomGatherAllToAllV, HcomGatherAllToAllVInferShape) {
    std::vector<string> dep_inputs = {"addrinfo", "recv_displacements", "recv_counts"};
    auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
    opDesc->SetOpInferDepends(dep_inputs);
    AttrUtils::SetBool(opDesc, "_force_unknown_shape", true);

    Tensor addrinfoTensor;
    Tensor recvDispTensor;
    Tensor recvCountsTensor;
    if ((op.GetInputConstData("addrinfo", addrinfoTensor) != GRAPH_SUCCESS) ||
        (op.GetInputConstData("recv_displacements", recvDispTensor) != GRAPH_SUCCESS) ||
        (op.GetInputConstData("recv_counts", recvCountsTensor) != GRAPH_SUCCESS)) {
        TensorDesc recvDataTensorDesc = op.GetOutputDesc("recv_data");
        recvDataTensorDesc.SetShape(ge::Shape(UNKNOWN_SHAPE));
        recvDataTensorDesc.SetDataType((DataType) op.get_attr_dtype());
        op.UpdateOutputDesc("recv_data", recvDataTensorDesc);
        TensorDesc gatherdTensorDesc = op.GetOutputDesc("gathered");
        gatherdTensorDesc.SetShape(ge::Shape(UNKNOWN_SHAPE));
        gatherdTensorDesc.SetDataType((DataType) op.get_attr_dtype());
        op.UpdateOutputDesc("gathered", gatherdTensorDesc);
        return GRAPH_SUCCESS;
    }

    DataType recvDispDtype = op.GetInputDesc("recv_displacements").GetDataType();
    vector<int64_t> recvDisp;
    GetConstValue(op, recvDispTensor, recvDispDtype, recvDisp);

    DataType recvCountsDtype = op.GetInputDesc("recv_counts").GetDataType();
    vector<int64_t> recvCounts;
    GetConstValue(op, recvCountsTensor, recvCountsDtype, recvCounts);

    std::vector<int64_t> recvDataDims;
    recvDataDims.push_back((int64_t)recvDisp.back() + (int64_t)recvCounts.back());
    TensorDesc outTensorDesc = op.GetOutputDesc("recv_data");
    outTensorDesc.SetShape(ge::Shape(recvDataDims));
    outTensorDesc.SetDataType(op.GetInputDesc("send_data").GetDataType());
    op.UpdateOutputDesc("recv_data", outTensorDesc);

    int64_t addrLen;
    int64_t addrLenAttr;
    if (op.GetAttr("addr_length", addrLenAttr) == GRAPH_SUCCESS) {
        const int64_t attrAddrLengthMin = -2;
        const int64_t attrAddrLengthSameLen = -2;
        const int32_t dataLenPerAddrinfo = 2;

        if (addrLenAttr < attrAddrLengthMin) {
            OP_LOGE(op.GetName().c_str(), "Attr attr_length [%lld] is not supported. expecttd: [%lld ~ ]",
                addrLenAttr, attrAddrLengthMin);
            return GRAPH_FAILED;
        } else if (addrLenAttr < 0) {
            DataType addrInfoDtype = op.GetInputDesc("addrinfo").GetDataType();
            vector<uint64_t> addrinfo;
            GetConstValue(op, addrinfoTensor, addrInfoDtype, addrinfo);
            if ((addrinfo.size() % dataLenPerAddrinfo) || (addrinfo.size() == 0)) {
                OP_LOGE(op.GetName().c_str(), "Input addrinfo is invalid, size is %zu.", addrinfo.size());
                return GRAPH_FAILED;
            }

            if (addrLenAttr == attrAddrLengthSameLen) {
                addrLen = addrinfo[1] * (addrinfo.size() / dataLenPerAddrinfo);
            } else {
                for (size_t i = 0; i < (addrinfo.size() / dataLenPerAddrinfo); i++) {
                    addrLen += addrinfo[i * dataLenPerAddrinfo + 1];
                }
            }
        } else {
            auto inShape = op.GetInputDesc("addrinfo").GetShape();
            std::vector<int64_t> inDims = inShape.GetDims();
            if (inDims.size() == 0) {
                OP_LOGE(op.GetName().c_str(), "Input addrinfo tensor's dims is invalid, expected: >0, actual:%zu.",
                    inDims.size());
                return GRAPH_FAILED;
            }
            if (inDims[0] <= 0) {
                OP_LOGE(op.GetName().c_str(), "Input addrinfo tensor's dim[0] is invalid, expected: >0, actual:%u.",
                    inDims[0]);
                return GRAPH_FAILED;
            }
            addrLen = inDims[0] / dataLenPerAddrinfo * addrLenAttr;
        }
    } else {
        OP_LOGE(op.GetName().c_str(), "Get attr addr_length failed.");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> gatheredDims;
    gatheredDims.push_back(addrLen);
    TensorDesc gatheredTensorDesc = op.GetOutputDesc("gathered");
    gatheredTensorDesc.SetShape(ge::Shape(gatheredDims));
    gatheredTensorDesc.SetDataType((DataType) op.get_attr_dtype());
    op.UpdateOutputDesc("gathered", gatheredTensorDesc);
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(HcomGatherAllToAllV, HcomGatherAllToAllVVerify) {
  // TODO：校验addrinfo_count_per_rank的维度
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(HcomGatherAllToAllV, HcomGatherAllToAllVInferShape);
VERIFY_FUNC_REG(HcomGatherAllToAllV, HcomGatherAllToAllVVerify);

}  // namespace ge

