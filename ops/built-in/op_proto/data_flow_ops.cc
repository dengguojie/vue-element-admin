/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file data_flow_ops.cpp
 * \brief
 */
#include "inc/data_flow_ops.h"
#include <utility>
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "error_util.h"
#include "securec.h"
#include <regex>

namespace ge {
static const int kRangeMaxNum = 2;
namespace {
graphStatus SetAttrsToShapesAndTypes(Operator& op,
                                     const std::string& dtypes,
                                     const std::string& shapes) {
  std::vector<DataType> elem_types;
  if (op.GetAttr(dtypes.c_str(), elem_types) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr [%s] failed.", dtypes.c_str());
    return GRAPH_FAILED;
  }
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get GetInferenceContext failed"));
    return GRAPH_FAILED;
  }

  Operator::OpListListInt elem_shapes;
  auto ret = op.GetAttr(shapes.c_str(), elem_shapes);
  OP_LOGI(TbeGetName(op).c_str(), "elem_shapes = %ld, elem_types = %ld.", elem_shapes.size(), elem_types.size());
  if (ret == GRAPH_SUCCESS && elem_shapes.size() > 0) {
    size_t num = std::min(elem_shapes.size(), elem_types.size());
    std::vector<ShapeAndType> handle_shapes_and_types;
    handle_shapes_and_types.reserve(num);

    for (size_t j = 0; j < num; ++j) {
      Shape elemShape(std::move(elem_shapes[j]));
      DataType elemType(std::move(elem_types[j]));
      ShapeAndType shape_and_type(elemShape, elemType);
      handle_shapes_and_types.emplace_back(std::move(shape_and_type));
    }

    std::vector<std::vector<ShapeAndType>> shapeAndTypes(2);
    shapeAndTypes[0] = handle_shapes_and_types;
    context->SetOutputHandleShapesAndTypes(shapeAndTypes);
  }

  AscendString op_name;
  op.GetName(op_name);
  std::vector<AscendString> marks = {op_name};
  context->SetMarks(marks);

  return GRAPH_SUCCESS;
}

graphStatus InferShapesFillUnknownShape(Operator& op, const std::string &name,
                                        const std::string &dynCompName, Shape &unknownShape) {
  std::vector<DataType> data_types;
  if (op.GetAttr(name.c_str(), data_types) == GRAPH_FAILED) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), name);
    return GRAPH_FAILED;
  }
  size_t size = data_types.size();
  for (size_t j = 0; j < size; j++) {
    TensorDesc output_desc = op.GetDynamicOutputDesc(dynCompName.c_str(), j);
    output_desc.SetShape(unknownShape);
    output_desc.SetDataType(data_types[j]);
    if (op.UpdateDynamicOutputDesc(dynCompName.c_str(), j, output_desc) != GRAPH_SUCCESS) {
      std::string errMsg = ConcatString("update desc of ", j, "th output of dynamic output[y] failed.");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

AicpuResourceContext* GetAicpuResourceContext(Operator& op, std::vector<AscendString> &marks) {
    auto context = op.GetInferenceContext();
    if (marks.empty() || context == nullptr) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("marks.empty()"));
      return nullptr;
    }

    if (context->RegisterReliedOnResourceKey(marks[0]) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("RegisterReliedOnResourceKey fail"));
      return nullptr;
    }

    return reinterpret_cast<AicpuResourceContext*>(context->GetResourceContext(marks[0]));
}

AicpuResourceContext* GetAicpuResourceContext(Operator& op) {
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    return nullptr;
  }
  std::vector<AscendString> marks;
  context->GetMarks(marks);
  return GetAicpuResourceContext(op, marks);
}

graphStatus DequeueManyShape(Operator& op, const Shape& n_shape, const std::string& out_name) {
  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> handleShapesAndTypes;
  handleShapesAndTypes = context->GetInputHandleShapesAndTypes();

  std::vector<ge::DataType> inputComponentTypes;
  if (op.GetAttr("component_types", inputComponentTypes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get attr[component_types] failed."));
    return GRAPH_FAILED;
  }

  graphStatus status = GRAPH_SUCCESS;
  size_t componentNum = inputComponentTypes.size();
  if ((handleShapesAndTypes.size() != 0) &&
      (handleShapesAndTypes[0].size() != 0) &&
      (handleShapesAndTypes[0].size() == componentNum)) {
    for (size_t j = 0; j < handleShapesAndTypes[0].size(); ++j) {
      Shape comibinedShape;
      Shape handleShape = handleShapesAndTypes[0][j].GetShape();
      graphStatus concatenateStatus = Concatenate(n_shape, handleShape, comibinedShape);
      if (concatenateStatus != GRAPH_SUCCESS) {
        std::string err_msg("call Concatenate function failed to cancate shape");
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
      TensorDesc outputDesc = op.GetDynamicOutputDesc(out_name.c_str(), j);
      outputDesc.SetShape(comibinedShape);
      outputDesc.SetDataType(inputComponentTypes[j]);
      status = op.UpdateDynamicOutputDesc(out_name.c_str(), j, outputDesc);
      if (status != GRAPH_SUCCESS) {
        std::string err_msg = ConcatString("update output[", out_name, ":", j, "] desc failed.");
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
    }
  } else {
    auto resContext = GetAicpuResourceContext(op);
    if (resContext == nullptr) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("marks is empty()"));
      Shape unknownShape(ge::UNKNOWN_SHAPE);
      InferShapesFillUnknownShape(op, "component_types", out_name, unknownShape);
      return GRAPH_SUCCESS;
    }
    std::string errMsg = ConcatString("size::", resContext->shape_and_range_.size(), componentNum);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
    if (resContext->shape_and_range_.size() == componentNum) {
      for (size_t j = 0; j < componentNum; j++) {
        Shape combinedShape;
        graphStatus ret = Concatenate(n_shape, resContext->shape_and_range_[j].shape_, combinedShape);
        if (ret != GRAPH_SUCCESS) {
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("call Concatenate failed"));
          return GRAPH_FAILED;
        }
        TensorDesc outputDesc = op.GetDynamicOutputDesc(out_name.c_str(), j);
        outputDesc.SetShape(combinedShape);
        outputDesc.SetDataType(inputComponentTypes[j]);
        status = op.UpdateDynamicOutputDesc(out_name.c_str(), j, outputDesc);
        if (status != GRAPH_SUCCESS) {
          std::string err_msg = ConcatString("update output[", out_name, ":", j, "] desc failed.");
          AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
          return GRAPH_FAILED;
        }
      }
    } else {
      Shape unknownShape(ge::UNKNOWN_RANK);
      InferShapesFillUnknownShape(op, "component_types", out_name, unknownShape);
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus GetStageKeyMarks(Operator &op, std::vector<AscendString> &marks) {
  std::string containerName;
  std::string sharedName;
  graphStatus resContainer;
  graphStatus resShar;
  resContainer = op.GetAttr("container", containerName);
  resShar = op.GetAttr("shared_name", sharedName);
  if ((resContainer != GRAPH_SUCCESS) || (resShar != GRAPH_SUCCESS)) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr failed."));
    return GRAPH_FAILED;
  }

  std::string indicesTensorName = containerName + sharedName;
  marks.push_back(AscendString(indicesTensorName.c_str()));
  return GRAPH_SUCCESS;
}

graphStatus InferShapesFromAicpuResource(Operator& op, std::vector<AscendString> &marks,
                                         const std::string &name, const std::string &dynCompName) {
  std::vector<DataType> dtypes;
  if (op.GetAttr(name.c_str(), dtypes) == GRAPH_FAILED) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), name);
    return GRAPH_FAILED;
  }

  auto context = GetAicpuResourceContext(op, marks);
  if (context == nullptr) {
    Shape unknown_shape(ge::UNKNOWN_RANK);
    return InferShapesFillUnknownShape(op, name, dynCompName, unknown_shape);
  }

  size_t size = dtypes.size();
  size_t num = std::min(context->shape_and_range_.size(), size);
  for (size_t j = 0; j < num; j++) {
    TensorDesc outputDesc = op.GetDynamicOutputDesc(dynCompName.c_str(), j);
    outputDesc.SetShape(context->shape_and_range_[j].shape_);
    outputDesc.SetDataType(dtypes[j]);
    if (op.UpdateDynamicOutputDesc(dynCompName.c_str(), j, outputDesc) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("update desc of ", j, "th output of dynamic output[y] failed.");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus SaveShapesToAicpuResource(const Operator &op, std::vector<AscendString> &marks,
                                      const size_t dynCompSize, const char*dynCompName) {
  auto context = op.GetInferenceContext();
  if (context == nullptr || marks.empty()) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("op.GetInferenceContext() fail"));
    return GRAPH_FAILED;
  }

  bool shapeChanged = false;
  auto resContext = reinterpret_cast<AicpuResourceContext*>(context->GetResourceContext(marks[0]));
  if (resContext == nullptr) {
    resContext = new (std::nothrow)AicpuResourceContext();
    if (resContext == nullptr) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("new AicpuResourceContext() fail"));
      return GRAPH_FAILED;
    }

    for (size_t j = 0; j < dynCompSize; j++) {
      const TensorDesc inputDesc = op.GetDynamicInputDesc(dynCompName, j);
      Shape elemShape(inputDesc.GetShape());
      std::vector<std::pair<int64_t, int64_t>> valueShapeRange;
      inputDesc.GetShapeRange(valueShapeRange);
      ShapeAndRange feed_shape_and_range{elemShape, valueShapeRange};
      resContext->shape_and_range_.push_back(feed_shape_and_range);
    }

    if (context->SetResourceContext(marks[0], resContext) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("SetResourceContext() fail"));
      delete resContext;
      return GRAPH_FAILED;
    }
    shapeChanged = true;
  } else {
    auto &shapeAndRange = resContext->shape_and_range_;
    std::vector<ShapeAndRange> newShapeAndRange;
    for (size_t j = 0; j < dynCompSize; j++) {
      const TensorDesc inputDesc = op.GetDynamicInputDesc(dynCompName, j);
      Shape elemShape(inputDesc.GetShape());
      std::vector<std::pair<int64_t, int64_t>> shapeRange;
      inputDesc.GetShapeRange(shapeRange);
      ShapeAndRange feedShapeAndRange{elemShape, shapeRange};
      newShapeAndRange.push_back(feedShapeAndRange);
    }

    if (newShapeAndRange.size() != shapeAndRange.size()) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("size() != shapeAndRange.size()"));
      return GRAPH_FAILED;
    }

    AscendString opName;
    op.GetName(opName);
    for (size_t j = 0; j < shapeAndRange.size(); j++) {
      if (MergeShapeAndRange(shapeAndRange[j], newShapeAndRange[j], shapeAndRange[j],
                             shapeChanged, opName.GetString()) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                           std::string("MergeShapeAndRange failed."));
        return GRAPH_FAILED;
      }
    }
  }

  if (shapeChanged) {
    if (context->AddChangedResourceKey(marks[0]) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("AddChangedResourceKey failed."));
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}
}  // namespace

IMPLEMT_INFERFUNC(QueueIsClosed, QueueIsClosedInfer) {
  Shape scalarShape;
  (void)Scalar(scalarShape);
  TensorDesc opsOutputDesc = op.GetOutputDescByName("is_closed");
  opsOutputDesc.SetShape(scalarShape);
  opsOutputDesc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("is_closed", opsOutputDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("UpdateOutputDesc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueIsClosed, QueueIsClosedInfer);

IMPLEMT_INFERFUNC(QueueSize, QueueSizeInfer) {
  Shape inputShape = op.GetInputDescByName("handle").GetShape();
  TensorDesc opsSizeDesc = op.GetOutputDescByName("size");
  opsSizeDesc.SetShape(inputShape);
  opsSizeDesc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", opsSizeDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("UpdateOutputDesc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueSize, QueueSizeInfer);

IMPLEMT_INFERFUNC(FIFOQueue, FIFOQueueInfer) {
  TensorDesc outputDesc = op.GetOutputDescByName("handle");
  DataType outputType = DT_RESOURCE;
  outputDesc.SetShape(Shape());
  outputDesc.SetDataType(outputType);
  if (op.UpdateOutputDesc("handle", outputDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("UpdateOutputDesc failed"));
    return GRAPH_FAILED;
  }

  return SetAttrsToShapesAndTypes(op, "component_types", "shapes");
}

INFER_FUNC_REG(FIFOQueue, FIFOQueueInfer);

IMPLEMT_INFERFUNC(QueueEnqueue, QueueEnqueueInfer) {
  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> handleShapesAndTypes;
  handleShapesAndTypes = context->GetInputHandleShapesAndTypes();
  size_t dynCompSize = op.GetInputsSize() - 1;
  if ((handleShapesAndTypes.size() != 0) &&
      (handleShapesAndTypes[0].size() != 0) &&
      (handleShapesAndTypes[0].size() == dynCompSize)) {
      OP_LOGI(TbeGetName(op).c_str(), "dynCompSize = %ld", dynCompSize);
    return GRAPH_SUCCESS;
  }

  std::vector<AscendString> marks;
  context->GetMarks(marks);
  OP_LOGI(TbeGetName(op).c_str(), "SaveShapesToAicpuResource marks.size() = %ld.", marks.size());
  return SaveShapesToAicpuResource(op, marks, dynCompSize, "components");
}

INFER_FUNC_REG(QueueEnqueue, QueueEnqueueInfer);

IMPLEMT_INFERFUNC(QueueEnqueueMany, QueueEnqueueManyInfer) {
  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> handleShapesAndTypes;
  handleShapesAndTypes = context->GetInputHandleShapesAndTypes();
  size_t dynCompSize = op.GetInputsSize() - 1;
  if ((handleShapesAndTypes.size() != 0) &&
      (handleShapesAndTypes[0].size() != 0) &&
      (handleShapesAndTypes[0].size() == dynCompSize)) {
    return GRAPH_SUCCESS;
  }

  std::vector<AscendString> marks;
  context->GetMarks(marks);
  return SaveShapesToAicpuResource(op, marks, dynCompSize, "components");
}

INFER_FUNC_REG(QueueEnqueueMany, QueueEnqueueManyInfer);

IMPLEMT_INFERFUNC(QueueDequeue, QueueDequeueInfer) {
  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> handleShapesAndTypes;
  handleShapesAndTypes = context->GetInputHandleShapesAndTypes();

  std::vector<ge::DataType> componentTypes;
  if (op.GetAttr("component_types", componentTypes) != GRAPH_SUCCESS) {
    std::string err_msg("get attr[component_types] failed.");
	  AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return 1;
  }

  graphStatus status = GRAPH_SUCCESS;
  size_t componentNum = componentTypes.size();
  if ((handleShapesAndTypes.size() != 0) &&
      (handleShapesAndTypes[0].size() != 0) &&
      (handleShapesAndTypes[0].size() == componentNum)) {
    for (size_t j = 0; j < handleShapesAndTypes[0].size(); ++j) {
      TensorDesc outputDesc = op.GetDynamicOutputDesc("components", j);
      outputDesc.SetShape(handleShapesAndTypes[0][j].GetShape());
      outputDesc.SetDataType(componentTypes[j]);
      status = op.UpdateDynamicOutputDesc("components", j, outputDesc);
      if (status != GRAPH_SUCCESS) {
        std::string err_msg = ConcatString("update output[components:", j, "] desc failed.");
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return 2;
      }
    }
    return GRAPH_SUCCESS;
  }

  std::vector<AscendString> marks;
  context->GetMarks(marks);
  return InferShapesFromAicpuResource(op, marks, "component_types", "components");
}

INFER_FUNC_REG(QueueDequeue, QueueDequeueInfer);

IMPLEMT_INFERFUNC(QueueDequeueMany, QueueDequeueManyInfer) {
  Tensor nTensor;
  Shape nShape;
  if (op.GetInputConstData("n", nTensor) == GRAPH_SUCCESS) {
    const uint8_t* n = nTensor.GetData();
    const int32_t* n_data = reinterpret_cast<const int32_t*>(n);
    if (*n_data < 0) {
      std::string err_msg = ConcatString("input[n] must >= 0, but got[", *n_data, "].");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    nShape = Shape({*n_data});
  } else {
    nShape = Shape({ge::UNKNOWN_DIM});
  }

  if (DequeueManyShape(op, nShape, "components") != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("DequeueManyShape failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueDequeueMany, QueueDequeueManyInfer);

IMPLEMT_INFERFUNC(QueueDequeueUpTo, QueueDequeueUpToInfer) {
  Shape n_shape({ge::UNKNOWN_DIM});
  return DequeueManyShape(op, n_shape, "components");
}

INFER_FUNC_REG(QueueDequeueUpTo, QueueDequeueUpToInfer);

IMPLEMT_INFERFUNC(Stage, StageInfer) {
  size_t size = op.GetInputsSize();

  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);
  return SaveShapesToAicpuResource(op, marks, size, "values");
}

INFER_FUNC_REG(Stage, StageInfer);

IMPLEMT_INFERFUNC(StageClear, StageClearInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StageClear, StageClearInfer);

IMPLEMT_INFERFUNC(StagePeek, StagePeekInfer) {
  Shape shape(ge::UNKNOWN_SHAPE);

  Operator::OpListType dtypes;
  if (op.GetAttr("dtypes", dtypes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get attr[dtypes] failed."));
    return GRAPH_FAILED;
  }

  size_t size = dtypes.size();
  for (size_t j = 0; j < size; ++j) {
    TensorDesc outputDesc = op.GetDynamicOutputDesc("y", j);
    outputDesc.SetShape(shape);
    outputDesc.SetDataType(dtypes[j]);
    op.UpdateDynamicOutputDesc("y", j, outputDesc);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StagePeek, StagePeekInfer);

IMPLEMT_INFERFUNC(StageSize, StageSizeInfer) {
  Shape outShape;
  (void)Scalar(outShape);

  TensorDesc output_desc = op.GetOutputDescByName("size");
  output_desc.SetShape(outShape);
  output_desc.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_desc);
}

INFER_FUNC_REG(StageSize, StageSizeInfer);

IMPLEMT_INFERFUNC(StackPop, StackPopInfer) {
  auto context = op.GetInferenceContext();
  Shape unknownRank(ge::UNKNOWN_RANK);

  OP_LOGI(TbeGetName(op).c_str(), "StackPopInfer.");
  DataType dataType;
  if (op.GetAttr("elem_type", dataType) != GRAPH_SUCCESS) {
    std::string err_msg("op get attr[elem_type] failed.");
	  AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc outputDesc = op.GetOutputDescByName("element");
  outputDesc.SetShape(unknownRank);
  std::vector<AscendString> marks;
  context->GetMarks(marks);
  if (marks.size() != 0) {
    ShapeAndRange shapeAndRange;
    bool geted = false;
    if (GetShapeAndRange(op, shapeAndRange, geted, context) != GRAPH_SUCCESS) {
	    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
         std::string("context is empty, you should call stack push first."));
      return GRAPH_FAILED;
    }

    if (geted) {
      outputDesc.SetShape(shapeAndRange.shape_);
      outputDesc.SetShapeRange(shapeAndRange.shape_range_);
    } else {
      OP_LOGW(TbeGetName(op).c_str(), "Stack is empty");
    }
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "Stack marks is empty");
  }

  outputDesc.SetDataType(dataType);
  return op.UpdateOutputDesc("element", outputDesc);
}

INFER_FUNC_REG(StackPop, StackPopInfer);

IMPLEMT_INFERFUNC(StackPush, StackPushInfer) {
  auto context = op.GetInferenceContext();
  Shape elShape = op.GetInputDescByName("element").GetShape();
  DataType type = op.GetInputDescByName("element").GetDataType();

  OP_LOGI(TbeGetName(op).c_str(), "StackPushInfer.");
  std::vector<AscendString> marks;
  context->GetMarks(marks);
  if (marks.size() != 0) {
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    op.GetInputDescByName("element").GetShapeRange(shapeRange);
    ShapeAndRange feedShapeAndRange{elShape, shapeRange, type};
    if (SetShapeAndRange(op, feedShapeAndRange) != GRAPH_SUCCESS) {
	    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("SetShapeAndRange failed"));
      return GRAPH_FAILED;
    }
  }

  TensorDesc opsOutput = op.GetOutputDescByName("y");
  opsOutput.SetShape(elShape);
  opsOutput.SetDataType(type);
  return op.UpdateOutputDesc("y", opsOutput);
}

INFER_FUNC_REG(StackPush, StackPushInfer);

IMPLEMT_INFERFUNC(StackClose, StackCloseInfer) {
  Shape shape;
  ge::TensorDesc handleDesc = op.GetInputDescByName("handle");
  if (WithRank(handleDesc, 1, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(handleDesc.GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StackClose, StackCloseInfer);

IMPLEMT_INFERFUNC(Stack, StackInfer) {
  Shape outShape;

  OP_LOGI(TbeGetName(op).c_str(), "StackInfer.");
  (void)Vector(2, outShape);
  auto context = op.GetInferenceContext();
  std::vector<std::string> marks = {TbeGetName(op)};

  context->SetMarks(marks);
  TensorDesc outputDesc = op.GetOutputDescByName("handle");
  outputDesc.SetShape(outShape);
  outputDesc.SetDataType(DT_RESOURCE);
  return op.UpdateOutputDesc("handle", outputDesc);
}

INFER_FUNC_REG(Stack, StackInfer);

IMPLEMT_INFERFUNC(MapClear, MapClearInfer) { return GRAPH_SUCCESS; }

INFER_FUNC_REG(MapClear, MapClearInfer);

IMPLEMT_INFERFUNC(MapIncompleteSize, MapIncompleteSizeInfer) {
  Shape scalarShape;
  (void)Scalar(scalarShape);
  TensorDesc outputDesc = op.GetOutputDescByName("size");
  outputDesc.SetShape(scalarShape);
  outputDesc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", outputDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("update description for output[size] failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MapIncompleteSize, MapIncompleteSizeInfer);

IMPLEMT_INFERFUNC(Unstage, UnstageInfer) {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);
  return InferShapesFromAicpuResource(op, marks, "dtypes", "y");
}

INFER_FUNC_REG(Unstage, UnstageInfer);

IMPLEMT_INFERFUNC(TensorArray, TensorArrayInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc handleDesc = op.GetOutputDescByName("handle");
  handleDesc.SetShape(Shape());
  handleDesc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", handleDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[handle] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc flowDesc = op.GetOutputDescByName("flow");
  flowDesc.SetShape(Shape());
  flowDesc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow", flowDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[flow] desc failed."));
    return GRAPH_FAILED;
  }

  bool identicalShapes;
  if (op.GetAttr("identical_element_shapes", identicalShapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[identical_element_shapes] failed."));
    return GRAPH_FAILED;
  }

  Operator::OpType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }

  Operator::OpListInt elemDims;
  if (op.GetAttr("element_shape", elemDims) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[element_shape] failed."));
    return GRAPH_FAILED;
  }

  AscendString op_name;
  op.GetName(op_name);
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        std::string(op_name.GetString()),
        std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }

  Shape elemShape(std::move(elemDims));
  if ((ShapeFullDefined(elemShape) || identicalShapes)) {
    ShapeAndType shape_and_type(elemShape, dtype);
    std::vector<ShapeAndType> handle_shapes_and_type;
    handle_shapes_and_type.reserve(1);
    handle_shapes_and_type.emplace_back(shape_and_type);
    std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
    shapes_and_types[0] = handle_shapes_and_type;
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }
  std::vector<AscendString> marks = {op_name};
  context->SetMarks(marks);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArray, TensorArrayInfer);

IMPLEMT_INFERFUNC(TensorArrayClose, TensorArrayCloseInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayClose, TensorArrayCloseInfer);

IMPLEMT_INFERFUNC(TensorArrayConcat, TensorArrayConcatInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Operator::OpType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc value_desc = op.GetOutputDescByName("value");
  value_desc.SetDataType(dtype);
  // unknown rank
  auto value_shape = Shape(ge::UNKNOWN_RANK);
  auto lengths_shape = Shape({ge::UNKNOWN_DIM});
  ShapeAndRange shape_and_range;
  bool geted = false;
  auto context = op.GetInferenceContext();
  AscendString op_name;
  op.GetName(op_name);
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        std::string(op_name.GetString()),
        std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }
  if (GetShapeAndRange(op, shape_and_range, geted, context) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), std::string("get shape and range failed."));
    return GRAPH_FAILED;
  }
  if (geted && RankKnown(shape_and_range.shape_)) {
    if (Concatenate(lengths_shape, shape_and_range.shape_, value_shape) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Concatenate function, 1th shape",
                                         DebugString(lengths_shape.GetDims()), 
                                         " of unknown shape can't concatenate respurce' shape",
                                         DebugString(shape_and_range.shape_.GetDims()));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), err_msg);
      return GRAPH_FAILED;
    }
    std::vector<std::pair<int64_t, int64_t>> value_shape_range = {{0, -1}};
    for (size_t j = 0; j < shape_and_range.shape_range_.size(); j++) {
      value_shape_range.push_back(shape_and_range.shape_range_[j]);
    }
    value_desc.SetShapeRange(value_shape_range);
  }
  value_desc.SetShape(value_shape);
  if (op.UpdateOutputDesc("value", value_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[value] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc lengths_desc = op.GetOutputDescByName("lengths");
  lengths_desc.SetDataType(ge::DT_INT64);
  // 1-D, unknown dim
  lengths_desc.SetShape(lengths_shape);
  if (op.UpdateOutputDesc("lengths", lengths_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[lengths] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayConcat, TensorArrayConcatInfer);

IMPLEMT_INFERFUNC(TensorArrayGather, TensorArrayGatherInfer) {
  Shape unused;
  Shape indices_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, indices_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }

  Shape input_or_attr_shape;
  ShapeAndRange shape_and_range;
  bool geted = false;
  if (GetShapeAndRange(op, shape_and_range, geted, context) != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), std::string("get shape and range failed."));
    return GRAPH_FAILED;
  }
  if (geted && RankKnown(shape_and_range.shape_)) {
    input_or_attr_shape = shape_and_range.shape_;
  } else {
    std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();
    if ((!shapes_and_types.empty()) && (!shapes_and_types.at(0).empty())) {
      input_or_attr_shape = shapes_and_types.at(0).at(0).GetShape();
    } else {
      Operator::OpListInt elem_dims;
      if (op.GetAttr("element_shape", elem_dims) != GRAPH_SUCCESS) {
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[element_shape] failed."));
        return GRAPH_FAILED;
      }
      input_or_attr_shape = Shape(std::move(elem_dims));
    }
  }

  Shape output_shape;
  if (Concatenate(indices_shape, input_or_attr_shape, output_shape) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Concatenate function, 1th shape",
                                       DebugString(indices_shape.GetDims()), 
                                       " of input[indices] can't concatenate context' shape",
                                       DebugString(input_or_attr_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Operator::OpType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc value_desc = op.GetOutputDescByName("value");
  value_desc.SetDataType(dtype);
  value_desc.SetShape(output_shape);
  if (geted && RankKnown(output_shape)) {
    std::vector<std::pair<int64_t, int64_t>> indices_shape_range;
    op.GetInputDesc(1).GetShapeRange(indices_shape_range);
    std::vector<std::pair<int64_t, int64_t>> value_shape_range;
    for (size_t j = 0; j < indices_shape_range.size(); j++) {
      value_shape_range.push_back(indices_shape_range[j]);
    }
    const std::vector<std::pair<int64_t, int64_t>> &shape_range = shape_and_range.shape_range_;
    for (size_t j = 0; j < shape_range.size(); j++) {
      value_shape_range.push_back(shape_range[j]);
    }
    value_desc.SetShapeRange(value_shape_range);
  }
  if (output_shape.GetDims().size() != shape_and_range.shape_range_.size()) {
    std::vector<std::pair<int64_t, int64_t>> final_value_shape_range;
    value_desc.SetShapeRange(final_value_shape_range);
  }
  if (op.UpdateOutputDesc("value", value_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[value] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayGather, TensorArrayGatherInfer);

IMPLEMT_INFERFUNC(TensorArrayGrad, TensorArrayGradInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc grad_handle_desc = op.GetOutputDescByName("grad_handle");
  grad_handle_desc.SetShape(Shape());
  grad_handle_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("grad_handle", grad_handle_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[grad_handle] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc flow_out = op.GetOutputDescByName("flow_out");
  flow_out.SetShape(Shape());
  flow_out.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", flow_out) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[flow_out] desc failed."));
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!input_shapes_and_types.empty()) && (!input_shapes_and_types.at(0).empty())) {
    ShapeAndType shape_and_type = input_shapes_and_types.at(0).at(0);
    std::vector<ShapeAndType> grad_shape_and_type;
    grad_shape_and_type.reserve(1);
    grad_shape_and_type.emplace_back(shape_and_type);
    std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
    shapes_and_types[0] = grad_shape_and_type;
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayGrad, TensorArrayGradInfer);

IMPLEMT_INFERFUNC(TensorArrayWrite, TensorArrayWriteInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();
  Shape value_shape = op.GetInputDesc(2).GetShape();

  if (!shapes_and_types.empty() && !shapes_and_types.at(0).empty()) {
    ShapeAndType shape_and_type = shapes_and_types.at(0).at(0);

    if (Merge(shape_and_type.GetShape(), value_shape, unused, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      std::string err_msg =
          ConcatString("failed to call Merge function, 2th shape",
                       DebugString(value_shape.GetDims()), " of input[value] can't merge context' shape",
                       DebugString(shape_and_type.GetShape().GetDims()));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  op.GetInputDesc(2).GetShapeRange(shape_range);
  ShapeAndRange feed_shape_and_range = {value_shape, shape_range, DT_FLOAT};
  if (SetShapeAndRange(op, feed_shape_and_range) != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), std::string("set shape and range failed."));
    return GRAPH_FAILED;
  }

  TensorDesc flow_out = op.GetOutputDescByName("flow_out");
  flow_out.SetShape(Shape());
  flow_out.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", flow_out) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[flow_out] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayWrite, TensorArrayWriteInfer);

IMPLEMT_INFERFUNC(TensorArrayGradWithShape, TensorArrayGradWithShapeInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!input_shapes_and_types.empty()) && (!input_shapes_and_types.at(0).empty())) {
    Shape input_shape = input_shapes_and_types.at(0).at(0).GetShape();
    DataType input_type = input_shapes_and_types.at(0).at(0).GetDataType();

    int64_t prepend_rank = op.GetInputDesc(2).GetShape().GetDim(0);
    std::vector<std::vector<ShapeAndType>> out_shape_and_type_list(2);
    std::vector<ShapeAndType> out_shapes_and_types;
    if (RankKnown(input_shape) && (prepend_rank != ge::UNKNOWN_DIM)) {
      size_t input_rank = input_shape.GetDimNum();
      std::vector<int64_t> dims;
      dims.reserve(input_rank + prepend_rank);
      for (int i = 0; i < prepend_rank; i++) {
        dims.push_back(ge::UNKNOWN_DIM);
      }

      for (size_t i = 0; i < input_rank; i++) {
        dims.push_back(input_shape.GetDim(i));
      }

      Shape out_shape(dims);
      ShapeAndType shape_and_type(out_shape, input_type);
      out_shapes_and_types.emplace_back(shape_and_type);
    } else {
      Shape unknown_shape(ge::UNKNOWN_SHAPE);
      ShapeAndType shape_and_type(unknown_shape, input_type);
      out_shapes_and_types.emplace_back(shape_and_type);
    }

    out_shape_and_type_list[0] = out_shapes_and_types;
    context->SetOutputHandleShapesAndTypes(out_shape_and_type_list);
  }

  TensorDesc grad_output_desc = op.GetOutputDescByName("grad_handle");
  grad_output_desc.SetShape(Shape());
  grad_output_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("grad_handle", grad_output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[grad_handle] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc flow_output_desc = op.GetOutputDescByName("flow_out");
  flow_output_desc.SetShape(Shape());
  flow_output_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", flow_output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[flow_out] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayGradWithShape, TensorArrayGradWithShapeInfer);

IMPLEMT_INFERFUNC(TensorArrayRead, TensorArrayReadInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();

  Operator::OpType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("get attr[dtype] failed."));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetDataType(dtype);

  ShapeAndRange shape_and_range;
  bool geted = false;
  if (GetShapeAndRange(op, shape_and_range, geted, context) != GRAPH_SUCCESS) {
    AscendString op_name;
    op.GetName(op_name);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), std::string("get shape and range failed."));
    return GRAPH_FAILED;
  }
  if (geted && RankKnown(shape_and_range.shape_)) {
    output_desc.SetShape(shape_and_range.shape_);
    output_desc.SetShapeRange(shape_and_range.shape_range_);
  } else if ((!shapes_and_types.empty()) && (!shapes_and_types.at(0).empty())) {
    Shape output_shape = shapes_and_types.at(0).at(0).GetShape();
    output_desc.SetShape(output_shape);
  }else {
    output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
  }

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayRead, TensorArrayReadInfer);

IMPLEMT_INFERFUNC(TensorArrayScatter, TensorArrayScatterInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  if (WithRank(op.GetInputDesc(1), 1, indices_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape value_shape;
  int indices_rank = indices_shape.GetDimNum();
  std::string err_msg;
  if (WithRankAtLeast(op.GetInputDesc(2), indices_rank, value_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("at least [", indices_rank, "D].");
    err_msg = GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), err_msg.c_str());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  for (int i = 0; i < indices_rank; i++) {
    if ((value_shape.GetDim(i) >= 0) && (indices_shape.GetDim(i) >= 0) &&
        (value_shape.GetDim(i) != indices_shape.GetDim(i))) {
      err_msg = ConcatString(i, "th dim[", value_shape.GetDim(i), "] of input[value]",
                             " must equal to that dim[", indices_shape.GetDim(i), "] of input[indices].");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("get context failed, context is nullptr."));
    return GRAPH_FAILED;
  }

  Shape fed_shape;
  AscendString op_name;
  op.GetName(op_name);
  if (SubShape(value_shape, 1, value_shape.GetDimNum(), 1, fed_shape, op_name.GetString()) != GRAPH_SUCCESS) {
    err_msg = ConcatString("failed to call SubShape function to subshape from 1 to ",
        value_shape.GetDimNum(), " of input[value] shape", DebugString(value_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), ConcatString("get ", err_msg));
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!shapes_and_types.empty()) && (!shapes_and_types.at(0).empty())) {
    Shape tensor_shape = shapes_and_types.at(0).at(0).GetShape();

    if (Merge(tensor_shape, fed_shape, fed_shape, op_name.GetString()) != GRAPH_SUCCESS) {
      std::string err_msg =
          ConcatString("failed to call Merge function, sub shape",
                       DebugString(fed_shape.GetDims()),
                       " of input[value] can't merge context' shape",
                       DebugString(tensor_shape.GetDims()));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> value_shape_range;
  op.GetInputDesc(2).GetShapeRange(value_shape_range);
  std::vector<std::pair<int64_t, int64_t>> feed_shape_range;
  for (size_t j = 1; j < value_shape_range.size(); ++j) {
    feed_shape_range.push_back(value_shape_range[j]);
  }
  ShapeAndRange feed_shape_and_range = {fed_shape, feed_shape_range, DT_FLOAT};
  if (SetShapeAndRange(op, feed_shape_and_range) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(std::string(op_name.GetString()), std::string("set shape and range failed."));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("flow_out");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[flow_out] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayScatter, TensorArrayScatterInfer);

IMPLEMT_INFERFUNC(TensorArraySplit, TensorArraySplitInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 1, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "1D");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDescByName("flow_out");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[flow_out] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArraySplit, TensorArraySplitInfer);

IMPLEMT_INFERFUNC(TensorArraySize, TensorArraySizeInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDescByName("size");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("update output[size] desc failed."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArraySize, TensorArraySizeInfer);

IMPLEMT_INFERFUNC(MapStage, MapStageInfer) {
  size_t dynCompSize = op.GetInputsSize() - 2;

  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);
  return SaveShapesToAicpuResource(op, marks, dynCompSize, "values");
}

INFER_FUNC_REG(MapStage, MapStageInfer);

IMPLEMT_INFERFUNC(MapUnstage, MapUnstageInfer)  {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);
  return InferShapesFromAicpuResource(op, marks, "dtypes", "values");
}

INFER_FUNC_REG(MapUnstage, MapUnstageInfer);

IMPLEMT_INFERFUNC(MapUnstageNoKey, MapUnstageNoKeyInfer) {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);
  return InferShapesFromAicpuResource(op, marks, "dtypes", "values");
}

INFER_FUNC_REG(MapUnstageNoKey, MapUnstageNoKeyInfer);

IMPLEMT_INFERFUNC(MapPeek, MapPeekInfer) {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);

  return InferShapesFromAicpuResource(op, marks, "dtypes", "values");
}

INFER_FUNC_REG(MapPeek, MapPeekInfer);

IMPLEMT_INFERFUNC(MapSize, MapSizeInfer) {
  Shape scalarShape;
  (void)Scalar(scalarShape);
  TensorDesc outputDesc = op.GetOutputDescByName("size");
  outputDesc.SetDataType(DT_INT32);
  outputDesc.SetShape(scalarShape);
  if (op.UpdateOutputDesc("size", outputDesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update description for output[size] failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MapSize, MapSizeInfer);

IMPLEMT_INFERFUNC(RandomShuffleQueue, RandomShuffleQueueInfer) {
  Shape scalarShape;
  (void)Scalar(scalarShape);
  TensorDesc outHandleDesc = op.GetOutputDescByName("handle");
  outHandleDesc.SetShape(scalarShape);
  outHandleDesc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", outHandleDesc) != GRAPH_SUCCESS) {
    std::string err_msg("update output[handle] desc failed.");
	  AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  
  return SetAttrsToShapesAndTypes(op, "component_types", "shapes") ;
}

INFER_FUNC_REG(RandomShuffleQueue, RandomShuffleQueueInfer);

IMPLEMT_INFERFUNC(DynamicPartition, DynamicPartitionInfer) {
  auto data_tensor = op.get_input_desc_x();
  auto partitions_tensor = op.get_input_desc_partitions();
  auto data_shape = data_tensor.GetShape();
  auto partitions_shape = partitions_tensor.GetShape();
  const auto rank_of_data = data_shape.GetDimNum();
  const auto rank_of_partitions = partitions_shape.GetDimNum();
  std::vector<int64_t> data_dims = data_shape.GetDims();
  std::vector<int64_t> partitions_dims = partitions_shape.GetDims();
  auto num_partitions = op.get_attr_num_partitions();

  if (partitions_shape.GetDims() == ge::UNKNOWN_SHAPE) {
    OP_LOGW(TbeGetName(op).c_str(), "input partition is unknown_shape !");
    Shape unknown_shape(ge::UNKNOWN_SHAPE);
    for (int j = 0; j < num_partitions; ++j) {
      TensorDesc y_tensor = op.GetDynamicOutputDesc("y", j);
      y_tensor.SetShape(unknown_shape);
      op.UpdateDynamicOutputDesc("y", j, y_tensor);
    }
    return GRAPH_SUCCESS;
  }
  auto result = WithRankAtLeast(data_tensor, rank_of_partitions, data_shape,
                                TbeGetName(op).c_str());
  if (result != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRankAtLeast function, the rank of intput[x] must >= intput[partitions], ",
        rank_of_data, " and ", rank_of_partitions);
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims;
  dims.reserve(std::max(rank_of_partitions, rank_of_data));
  dims.resize(rank_of_partitions);
  for (uint8_t i = 0; i < rank_of_partitions; i++) {
    if (Merge(data_dims[i], partitions_dims[i], dims[i]) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function to merge the 0th input[x]'s dim",
          "[", data_dims[i], "]", "and the 1st intput[partitions]'s dim", "[" , partitions_dims[i], "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  vector<int64_t> output_dim0({ge::UNKNOWN_DIM});
  Shape output_unknown_dim0 = Shape(output_dim0);
  Shape data_suffix_shape;
  result = SubShape(data_shape, rank_of_partitions, rank_of_data, 1,
                    data_suffix_shape, TbeGetName(op).c_str());
  if (result != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call SubShape function to subshape, the 0th input[x]'s shape",
                     DebugString(data_dims), " and the 1st input[partitions]'s shape", DebugString(partitions_dims));
      return GRAPH_FAILED;
  }
  Shape output_shape;
  result = Concatenate(output_unknown_dim0, data_suffix_shape, output_shape);
  if (result != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Concatenate function to concatenate data suffix shape, "
        "which is from function SubShape result, it's shape",
        DebugString(data_suffix_shape.GetDims()), " and unkonw shape",
        DebugString(output_unknown_dim0.GetDims()));
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  for (int64_t j = 0; j < num_partitions; ++j) {
    TensorDesc y_tensor = op.GetDynamicOutputDesc("y", j);
    y_tensor.SetDataType(data_tensor.GetDataType());
    y_tensor.SetShape(output_shape);
    if (op.UpdateDynamicOutputDesc("y", j, y_tensor) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("update desc of ", j, "th output of dynamic output[y] failed.");
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicPartition, DynamicPartitionInfer);

IMPLEMT_INFERFUNC(DynamicStitch, DynamicStitchInfer) {
  auto num_incides = op.get_attr_N();
  if (num_incides < 1) {
    std::string err_msg =
        ConcatString("invalid value", "[", num_incides, "]",
                     " of attr[N], it should be not less than 1");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  // unknown shape support
  std::vector<std::string> input_infer_depends;
  for (int32_t j = 0; j < num_incides; ++j) {
    input_infer_depends.push_back(std::string("indices") + std::to_string(j));
  }
  op_desc->SetOpInferDepends(input_infer_depends);

  bool all_indices_constant = true;
  int32_t max_index = 0;
  int32_t zero_shape_num = 0;
  GeShape last_suffix_shape(ge::UNKNOWN_RANK);

  for (int64_t i = 0; i < num_incides; ++i) {
    auto indices_tensor_name = "indices" + std::to_string(i);
    auto data_tensor_name = "x" + std::to_string(i);

    Tensor unused_tensor;
    if (op.GetInputConstData(indices_tensor_name.c_str(), unused_tensor) != GRAPH_SUCCESS) {
      OP_LOGW(TbeGetName(op).c_str(), "try get indices %ld failed .", i);
      all_indices_constant = false;
    }

    auto indices_desc = op_desc->MutableInputDesc(indices_tensor_name);
    auto data_desc = op_desc->MutableInputDesc(data_tensor_name);

    const auto indices_shape = indices_desc->GetShape();
    const auto data_shape = data_desc->GetShape();

    auto indices_dims = indices_shape.GetDims();
    auto data_dims = data_shape.GetDims();
    if (indices_dims == UNKNOWN_RANK || data_dims == UNKNOWN_RANK) {
      continue;
    }
    if ((!indices_dims.empty()) && (!data_dims.empty()) && ((indices_dims[0] == 0) || (data_dims[0] == 0))) {
      zero_shape_num++;
      continue;
    }
    const auto rank_of_indices = indices_shape.GetDimNum();
    const auto rank_of_data = data_shape.GetDimNum();
    if (rank_of_indices > rank_of_data) {
      std::string err_msg = ConcatString(
          "the 0th intput[indices]'s rank must > the 1st intput[x], ",
          rank_of_indices, " and ", rank_of_data);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    std::vector<int64_t> dims;
    dims.reserve(std::max(rank_of_indices, rank_of_data));
    dims.resize(rank_of_indices);

    for (uint8_t j = 0; j < rank_of_indices; ++j) {
      if (data_dims[j] != UNKNOWN_DIM && indices_dims[j] != UNKNOWN_DIM &&
          (Merge(data_dims[j], indices_dims[j], dims[j]) != GRAPH_SUCCESS)) {
      std::string err_msg = ConcatString("failed to call Merge function to merge the 0th input[indices]'s dim",
          "[", indices_dims[j], "]", "and the 1st input[indices]'s dim", "[", data_dims[j], "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
    }

    GeShape data_suffix_shape;
    if (SubShape(data_shape, rank_of_indices, rank_of_data, 1, data_suffix_shape,
                 TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call SubShape function, the 1th input[x]'s shape",
                     DebugString(data_dims));
      return GRAPH_FAILED;
    }

    if (Merge(last_suffix_shape, data_suffix_shape, last_suffix_shape,
              TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      std::string err_msg =
          ConcatString("failed to call Merge function to merge unknow shape",
                       DebugString(last_suffix_shape.GetDims()), " and 1th input[x]'s SubShape",
                       DebugString(data_suffix_shape.GetDims()));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    Tensor indices_data_tensor;
    if (op.GetInputConstData(indices_tensor_name.c_str(), indices_data_tensor) == GRAPH_SUCCESS) {
      const int32_t* indices_data = reinterpret_cast<const int32_t*>(indices_data_tensor.GetData());
      int64_t count = indices_shape.GetShapeSize();
      for (int32_t i = 0; i < count; ++i) {
        if (static_cast<int64_t>(indices_data[i]) > max_index) {
          max_index = static_cast<int64_t>(indices_data[i]);
        }
      }
    }
  }
  if (zero_shape_num == num_incides) {
    GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
    GeShape y_shape({ 0 });
    DataType y_data_type = op_desc->MutableInputDesc("x0")->GetDataType();
    (void)FillOpDesc(y_desc, y_shape, y_data_type);
    return GRAPH_SUCCESS;
  }
  auto output_dim0 = all_indices_constant ? (max_index + 1) : (ge::UNKNOWN_DIM);
  GeShape output_shape_prefix({output_dim0});
  GeShape output_shape;
  if (Concatenate(output_shape_prefix, last_suffix_shape, output_shape) != GRAPH_SUCCESS) {
      std:: string err_msg = ConcatString("failed to call Concatenate function to concatenate, prefix output shape",
          DebugString(output_shape_prefix.GetDims()),
          " and last suffix shape", DebugString(last_suffix_shape.GetDims()));
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
  GeShape y_shape(output_shape.GetDims());
  DataType y_data_type = op_desc->MutableInputDesc("x0")->GetDataType();

  (void)FillOpDesc(y_desc, y_shape, y_data_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicStitch, DynamicStitchInfer);

IMPLEMT_INFERFUNC(ParallelDynamicStitch, ParallelDynamicStitchInfer) {
  auto num_incides = op.get_attr_N();
  bool all_indices_constant = true;
  int32_t max_index = 0;
  Shape last_suffix_shape(ge::UNKNOWN_RANK);

  if (num_incides < 1) {
    std::string err_msg =
        ConcatString("invalid value", "[", num_incides, "]",
                     " of attr[N], it should be not less than 1");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.GetOutputDescByName("y");
  for (int64_t i = 0; i < num_incides; ++i) {
    auto indices_tensor_name = "indices" + std::to_string(i);

    std::vector<std::string> input_infer_depends = {indices_tensor_name};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(input_infer_depends);

    TensorDesc unused_tensor;
    const TensorDesc indices_tensor = op.GetDynamicInputDesc("indices", i);
    const TensorDesc data_tensor = op.GetDynamicInputDesc("x", i);
    const auto indices_shape = indices_tensor.GetShape();
    const auto data_shape = data_tensor.GetShape();
    if (!RankKnown(indices_shape)) {
      continue;
    }
    const auto rank_of_indices = indices_shape.GetDimNum();
    const auto rank_of_data = data_shape.GetDimNum();
    std::vector<int64_t> indices_dims = indices_shape.GetDims();
    std::vector<int64_t> data_dims = data_shape.GetDims();
    std::vector<int64_t> dims;
    dims.reserve(std::max(rank_of_indices, rank_of_data));
    dims.resize(rank_of_indices);

    for (uint8_t i = 0; i < rank_of_indices; ++i) {
      if ((Merge(data_dims[i], indices_dims[i], dims[i])) != GRAPH_SUCCESS) {
        std::string err_msg =
        ConcatString("failed to call Merge function to merge", i,
          "th data_dims", DebugString(data_shape.GetDims()),
          " and indices_dims", DebugString(indices_shape.GetDims()));
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
    }

    Shape data_suffix_shape;
    auto result = SubShape(data_shape, rank_of_indices, rank_of_data, 1, data_suffix_shape, TbeGetName(op).c_str());
    std::string  subshape_err_msg = ConcatString("failed  to subshape from 1 to ",
      data_shape.GetDimNum(), " of input[x] shape", DebugString(data_shape.GetDims()));
    if (result != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), ConcatString("get ", subshape_err_msg));
      return GRAPH_FAILED;
    }
    result = Merge(last_suffix_shape, data_suffix_shape, last_suffix_shape, TbeGetName(op).c_str());
    if (result != GRAPH_SUCCESS) {
      std::string err_msg =
          ConcatString("failed to call Merge function to merge unknow shape",
                       DebugString(last_suffix_shape.GetDims()), " and 1th input[x]'s SubShape",
                       DebugString(data_suffix_shape.GetDims()));
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    if (op.TryGetInputDesc(indices_tensor_name.c_str(), unused_tensor) == GRAPH_SUCCESS) {
      Tensor indices_data_tensor;
      result = op.GetInputConstData(indices_tensor_name.c_str(), indices_data_tensor);
      if (result == GRAPH_SUCCESS) {
        const int32_t* indices_data = reinterpret_cast<const int32_t*>(indices_data_tensor.GetData());
        if (indices_data != nullptr) {
          int64_t count = indices_data_tensor.GetTensorDesc().GetShape().GetShapeSize();
          for (int32_t i = 0; i < count; ++i) {
            if (static_cast<int64_t>(indices_data[i]) > max_index) {
              max_index = static_cast<int64_t>(indices_data[i]);
            }
          }
        } else {
          all_indices_constant = false;
        }
      } else {
        Shape output_shape(ge::UNKNOWN_RANK);
        auto data_tensor = op.GetDynamicInputDesc("x", 0);
        y_desc.SetDataType(data_tensor.GetDataType());
        y_desc.SetShape(output_shape);
        op.UpdateOutputDesc("y", y_desc);
        return GRAPH_SUCCESS;
      }
    }
  }

  auto output_dim0 = all_indices_constant ? (max_index) : (ge::UNKNOWN_DIM);
  Shape output_shapePrefix({output_dim0});
  Shape output_shape;
  auto result = Concatenate(output_shapePrefix, last_suffix_shape, output_shape);
  if (result != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Concatenate function to concatenate, prefix output shape",
      DebugString(output_shapePrefix.GetDims()), " and last suffix shape", DebugString(last_suffix_shape.GetDims()));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto data_tensor = op.GetDynamicInputDesc("x", 0);
  y_desc.SetDataType(data_tensor.GetDataType());
  y_desc.SetShape(output_shape);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("fail to update output[y] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ParallelDynamicStitch, ParallelDynamicStitchInfer);

IMPLEMT_INFERFUNC(PaddingFIFOQueue, PaddingFIFOQueueInfer) {
  TensorDesc outputDesc = op.GetOutputDescByName("handle");
  outputDesc.SetShape(Shape());
  outputDesc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", outputDesc) != GRAPH_SUCCESS) {
    std::string err_msg("update output[handle] desc failed.");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  
  return SetAttrsToShapesAndTypes(op, "component_types", "shapes") ;
}

INFER_FUNC_REG(PaddingFIFOQueue, PaddingFIFOQueueInfer);

IMPLEMT_INFERFUNC(PriorityQueue, PriorityQueueInfer) {
  TensorDesc outputDesc = op.GetOutputDescByName("handle");
  OP_LOGI(TbeGetName(op).c_str(), "PriorityQueueInfer.");

  outputDesc.SetShape(Shape());
  outputDesc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", outputDesc) != GRAPH_SUCCESS) {
    std::string err_msg("update output[handle] desc failed.");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  return SetAttrsToShapesAndTypes(op, "component_types", "shapes");
}

INFER_FUNC_REG(PriorityQueue, PriorityQueueInfer);

IMPLEMT_INFERFUNC(QueueClose, QueueCloseInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueClose, QueueCloseInfer);

IMPLEMT_INFERFUNC(OrderedMapStage, OrderedMapStageInfer) {
  size_t dynCompSize;

  if (op.GetInputsSize() < 2) {
    std::string err_msg("update output[handle] desc failed.");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  dynCompSize = op.GetInputsSize() - 2;
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);
  return SaveShapesToAicpuResource(op, marks, dynCompSize, "values");
}

INFER_FUNC_REG(OrderedMapStage, OrderedMapStageInfer);

IMPLEMT_INFERFUNC(OrderedMapSize, OrderedMapSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);

  TensorDesc output_tensor = op.GetOutputDescByName("size");
  output_tensor.SetShape(scalar_shape);
  output_tensor.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_tensor);
}

INFER_FUNC_REG(OrderedMapSize, OrderedMapSizeInfer);

IMPLEMT_INFERFUNC(OrderedMapClear, OrderedMapClearInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapClear, OrderedMapClearInfer);

IMPLEMT_INFERFUNC(OrderedMapIncompleteSize, OrderedMapIncompleteSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);

  TensorDesc output_tensor = op.GetOutputDescByName("size");
  output_tensor.SetShape(scalar_shape);
  output_tensor.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_tensor);
}

INFER_FUNC_REG(OrderedMapIncompleteSize, OrderedMapIncompleteSizeInfer);

IMPLEMT_INFERFUNC(OrderedMapPeek, OrderedMapPeekInfer) {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);

  return InferShapesFromAicpuResource(op, marks, "dtypes", "values");
}

INFER_FUNC_REG(OrderedMapPeek, OrderedMapPeekInfer);

IMPLEMT_INFERFUNC(OrderedMapUnstageNoKey, OrderedMapUnstageNoKeyInfer) {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);

  if (InferShapesFromAicpuResource(op, marks, "dtypes", "values") != GRAPH_SUCCESS) {
	  AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
      std::string("InferShapesFromAicpuResource failed"));
    return GRAPH_FAILED;
  }

  TensorDesc tensordescOutput = op.GetOutputDescByName("key");
  tensordescOutput.SetDataType(DT_INT64);
  (void)op.UpdateOutputDesc("key", tensordescOutput);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapUnstageNoKey, OrderedMapUnstageNoKeyInfer);

IMPLEMT_INFERFUNC(OrderedMapUnstage, OrderedMapUnstageInfer) {
  std::vector<AscendString> marks;
  GetStageKeyMarks(op, marks);

  return InferShapesFromAicpuResource(op, marks, "dtypes", "values");
}

INFER_FUNC_REG(OrderedMapUnstage, OrderedMapUnstageInfer);

IMPLEMT_INFERFUNC(Barrier, BarrierInfer) {
  std::vector<ge::DataType> component_types;
  if (op.GetAttr("component_types", component_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("failed to get attr[component_types]."));
    return GRAPH_FAILED;
  }
  if (component_types.size() < 1) {
    string err_msg = ConcatString(
        "the length of attr[component_types] should not be less than 1, ",
        "but get ", component_types.size(), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape out_shape;
  if (Vector(2, out_shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        string("failed to create vector."));
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDescByName("handle");
  output_desc.SetDataType(DT_STRING_REF);
  output_desc.SetShape(out_shape);
  op.UpdateOutputDesc("handle", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Barrier, BarrierInfer);

IMPLEMT_INFERFUNC(BarrierInsertMany, BarrierInsertManyInfer) {
  TensorDesc keys_desc = op.GetInputDescByName("keys");
  TensorDesc values_desc = op.GetInputDescByName("values");
  TensorDesc handle_desc = op.GetInputDescByName("handle");
  Shape handle_shape, keys_shape, values_shape;
  string err_msg;
  if (WithRank(handle_desc, 1, handle_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      err_msg = ConcatString(
          "failed to call WithRank function, the rank of input[handle] should be 1, ",
          "but get ", handle_desc.GetShape().GetDimNum());
    return GRAPH_FAILED;
  }
  int64_t unused_dim;
  if (WithValue(handle_shape.GetDim(0), 2, unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithValue function, dim[0] of input[handle] should be 2, ",
        "but get ", handle_shape.GetDim(0));
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(keys_desc, 1, keys_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, the rank of input[keys] should be 1, ",
        "but get ", keys_desc.GetShape().GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtLeast(values_desc, 1, values_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRankAtLeast function, ",
        "the rank of input[values] should be at least 1, but get ",
        values_desc.GetShape().GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  graphStatus status;
  Shape values_dim_zero_shape;
  Vector(values_shape.GetDim(0), values_dim_zero_shape);
  status = Merge(keys_shape, values_dim_zero_shape, handle_shape, TbeGetName(op).c_str());
  if (status != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call Merge function, can't merge the shape of input[keys]'s and ",
        "the shape whose value is equal to dim[0] of input[values].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierInsertMany, BarrierInsertManyInfer);

IMPLEMT_INFERFUNC(BarrierTakeMany, BarrierTakeManyInfer) {
  std::vector<ge::DataType> component_types;
  if (op.GetAttr("component_types", component_types) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        string("failed to get attr[component_types]"));
    return GRAPH_FAILED;
  }
  if (component_types.size() < 1) {
    string err_msg = ConcatString(
        "the length of attr[component_types] should not be less than 1, but get ",
        component_types.size(), ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  graphStatus status;
  TensorDesc output_desc;
  // update ordinary outputs as unknown shape
  vector<string> outputs_name = {"indices", "keys"};
  vector<DataType> outputs_type = {DT_INT64, DT_STRING};
  for (size_t i = 0; i < outputs_name.size(); ++i) {
    output_desc = op.GetOutputDescByName(outputs_name[i].c_str());
    output_desc.SetDataType(outputs_type[i]);
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    status = op.UpdateOutputDesc(outputs_name[i].c_str(), output_desc);
    if (status != GRAPH_SUCCESS) {
      string error_msg = ConcatString(
          "failed to update output[" , outputs_name[i], "] desc");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
  }
  // update dynamic outputs as unknown shape
  int size = op.GetDynamicOutputNum("values");
  for (int i = 0; i < size; ++i) {
    output_desc = op.GetDynamicOutputDesc("values", i);
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    output_desc.SetDataType(component_types[0]);
    status = op.UpdateDynamicOutputDesc("values", i, output_desc);
    if (status != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          std::string("failed to update output[values] desc."));
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierTakeMany, BarrierTakeManyInfer);

IMPLEMT_INFERFUNC(BarrierClose, BarrierCloseInfer) {
  Shape shape;
  int64_t unused_dim;
  string error_msg;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    if (WithRank(op.GetInputDesc(i), 1, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      error_msg = ConcatString(
        "failed to call WithRank function, the rank of input[", i,
        "] should be 1, but get ",
        op.GetInputDesc(i).GetShape().GetDimNum(), ".");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    if (WithValue(shape.GetDim(0), 2, unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      error_msg = ConcatString(
        "failed to call WithValue function, dim[0] of handle should be 2, but get ",
        shape.GetDim(0), ".");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierClose, BarrierCloseInfer);

IMPLEMT_INFERFUNC(BarrierReadySize, BarrierReadySizeInfer) {
  Shape shape;
  int64_t unused_dim;
  string error_msg;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    if (WithRank(op.GetInputDesc(i), 1, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      error_msg = ConcatString(
          "failed to call WithRank function, the rank of input[", i,
          "] should be 1, but get ",
          op.GetInputDesc(i).GetShape().GetDimNum(), ".");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    if (WithValue(shape.GetDim(0), 2, unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      error_msg = ConcatString(
          "failed to call WithValue function, the dim[0] of input[",
          i, "] should be 2, but get ", shape.GetDim(0), ".");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
  }
  TensorDesc output_desc = op.GetOutputDescByName("size");
  output_desc.SetDataType(DT_INT32);
  output_desc.SetShape(Shape());
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        string("failed to update output[size] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierReadySize, BarrierReadySizeInfer);

IMPLEMT_INFERFUNC(BarrierIncompleteSize, BarrierIncompleteSizeInfer) {
  Shape shape;
  int64_t unused_dim;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    if (WithRank(op.GetInputDesc(i), 1, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      string error_msg = ConcatString(
          "failed to call WithRank function, the rank of input[", i,
          "] should be 1, but get ",
          op.GetInputDesc(i).GetShape().GetDimNum(), ".");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    if (WithValue(shape.GetDim(0), 2, unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
      string error_msg = ConcatString(
          "failed to call WithValue function, dim[0] of input[",
          i, "] should be 2, but get ", shape.GetDim(0), ".");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
  }
  TensorDesc output_desc = op.GetOutputDescByName("size");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        string("failed to update output[size] desc."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierIncompleteSize, BarrierIncompleteSizeInfer);

IMPLEMT_INFERFUNC(RecordInput, RecordInputInfer) {
  graphStatus status;
  TensorDesc output_desc = op.GetOutputDescByName("records");
  output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  output_desc.SetDataType(DT_STRING);
  status = op.UpdateOutputDesc("records", output_desc);
  if (status != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("update output[records] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RecordInput, RecordInputInfer);

IMPLEMT_INFERFUNC(ConditionalAccumulator, ConditionalAccumulatorInfer) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get attr[type] failed"));
    return GRAPH_FAILED;
  }
  Shape output_shape;
  (void)Vector(2, output_shape);
  TensorDesc handle_desc = op.get_output_desc_handle();
  handle_desc.SetShape(output_shape);
  handle_desc.SetDataType(DT_STRING_REF);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("update output[handle] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ConditionalAccumulator, ConditionalAccumulatorInfer);

IMPLEMT_INFERFUNC(AccumulatorApplyGradient, AccumulatorApplyGradientInfer) {
  Shape unused_shape;
  auto local_step_tensor = op.get_input_desc_local_step();
  if (WithRank(local_step_tensor, 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
    DebugString(local_step_tensor.GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorApplyGradient, AccumulatorApplyGradientInfer);

IMPLEMT_INFERFUNC(AccumulatorNumAccumulated, AccumulatorNumAccumulatedInfer) {
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call function Scalar to create a scalar shape");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.get_output_desc_y();
  y_desc.SetShape(shape);
  y_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorNumAccumulated, AccumulatorNumAccumulatedInfer);

IMPLEMT_INFERFUNC(AccumulatorSetGlobalStep, AccumulatorSetGlobalStepInfer) {
  Shape shape;
  auto new_global_step_tensor = op.get_input_desc_new_global_step();
  if (WithRank(new_global_step_tensor, 0, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
    DebugString(new_global_step_tensor.GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorSetGlobalStep, AccumulatorSetGlobalStepInfer);

IMPLEMT_INFERFUNC(AccumulatorTakeGradient, AccumulatorTakeGradientInfer) {
  Shape shape;
  auto num_required_tensor = op.get_input_desc_num_required();
  if (WithRank(num_required_tensor, 0, shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
    DebugString(num_required_tensor.GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.get_output_desc_y();
  y_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  y_desc.SetDataType(dtype);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorTakeGradient, AccumulatorTakeGradientInfer);

IMPLEMT_INFERFUNC(SparseConditionalAccumulator, SparseConditionalAccumulatorInfer) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  Shape output_shape;
  if (Vector(2, output_shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        TbeGetName(op),
        string("call Vector function failed to create output shape"));
    return GRAPH_FAILED;
  }
  TensorDesc handle_desc = op.get_output_desc_handle();
  handle_desc.SetShape(output_shape);
  handle_desc.SetDataType(DT_STRING_REF);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[handle] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseConditionalAccumulator, SparseConditionalAccumulatorInfer);

IMPLEMT_INFERFUNC(SparseAccumulatorApplyGradient, SparseAccumulatorApplyGradientInfer) {
  Shape unused_shape;
  auto local_step_tensor = op.get_input_desc_local_step();
  if (WithRank(local_step_tensor, 0, unused_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[local_step] rank must "
        "be 0-D, got rank[",
        local_step_tensor.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseAccumulatorApplyGradient, SparseAccumulatorApplyGradientInfer);

IMPLEMT_INFERFUNC(SparseAccumulatorTakeGradient, SparseAccumulatorTakeGradientInfer) {
  Shape unused_shape;
  auto num_required_tensor = op.get_input_desc_num_required();
  if (WithRank(num_required_tensor, 0, unused_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[num_required] rank must "
        "be 0-D, got rank[",
        num_required_tensor.GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc indices_desc = op.get_output_desc_indices();
  indices_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[indices] desc failed"));
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc values_desc = op.get_output_desc_values();
  values_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  values_desc.SetDataType(dtype);
  if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[values] desc failed"));
    return GRAPH_FAILED;
  }

  TensorDesc shape_desc = op.get_output_desc_shape();
  shape_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[shape] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseAccumulatorTakeGradient, SparseAccumulatorTakeGradientInfer);

IMPLEMT_INFERFUNC(ResourceConditionalAccumulator, ResourceConditionalAccumulatorInfer) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        TbeGetName(op), std::string("get attr[dtype] failed"));
    return GRAPH_FAILED;
  }
  Shape vector_shape;
  // Set Output as Vector(2) of DT_RESOURCE
  if (Vector(2, vector_shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
      std::string("call Vector function create shape with dim 2 failed"));
    return GRAPH_FAILED;
  }
  auto handle_desc = op.GetOutputDescByName("handle");
  handle_desc.SetShape(vector_shape);
  handle_desc.SetDataType(DT_RESOURCE);

  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("update output[handle] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceConditionalAccumulator, ResourceConditionalAccumulatorInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorApplyGradient, ResourceAccumulatorApplyGradientInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDescByName("local_step"), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("input[local_step] has wrong shape",
      DebugString(op.GetInputDescByName("local_step").GetShape().GetDims()),
      ", it should be scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorApplyGradient, ResourceAccumulatorApplyGradientInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorNumAccumulated, ResourceAccumulatorNumAccumulatedInfer) {
  Shape scalarShape;
  if (Scalar(scalarShape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
        TbeGetName(op), std::string("call Scalar function create shape failed"));
    return GRAPH_FAILED;
  }
  auto num_accumulated_sesc = op.GetOutputDescByName("num_accumulated");
  num_accumulated_sesc.SetShape(scalarShape);
  num_accumulated_sesc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("num_accumulated", num_accumulated_sesc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("update output[num_accumulated] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorNumAccumulated, ResourceAccumulatorNumAccumulatedInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorSetGlobalStep, ResourceAccumulatorSetGlobalStepInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDescByName("new_global_step"), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("input[new_global_step] has wrong shape",
      DebugString(op.GetInputDescByName("new_global_step").GetShape().GetDims()),
      ", it should be scalar");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorSetGlobalStep, ResourceAccumulatorSetGlobalStepInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorTakeGradient,
                  ResourceAccumulatorTakeGradientInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDescByName("num_required"), 0, unused_shape,
               TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        2, DebugString(op.GetInputDescByName("num_required").GetShape().GetDims()),
        "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("failed to get attr[dtype]."));
    return GRAPH_FAILED;
  }
  auto average_desc = op.GetOutputDescByName("average");
  average_desc.SetShape(Shape(ge::UNKNOWN_RANK));
  average_desc.SetDataType(dtype);
  if (op.UpdateOutputDesc("average", average_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("update output[average] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorTakeGradient,
               ResourceAccumulatorTakeGradientInfer);

IMPLEMT_INFERFUNC(OutfeedEnqueueOp, OutfeedEnqueueInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OutfeedEnqueueOp, OutfeedEnqueueInfer);

IMPLEMT_INFERFUNC(OutfeedEnqueueOpV2, OutfeedEnqueueOpV2Infer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OutfeedEnqueueOpV2, OutfeedEnqueueOpV2Infer);

int32_t Split(const std::string &s, std::vector<std::string> &result, const char *delchar) {
  if (s.empty()) { return 0; }
  result.clear();
  char *buffer = new (std::nothrow) char[s.size() + 1];
  if (buffer == nullptr) {
    return -1;
  }
  buffer[s.size()] = '\0';
  errno_t e = strcpy_s(buffer, s.size() + 1, s.c_str());
  if (e != EOK) {
    delete[] buffer;
    return -1;
  }
  char *p_tmp = nullptr;
  char *p = strtok_s(buffer, delchar, &p_tmp);
  if (p != nullptr) {
    do { result.emplace_back(p); } while ((p = strtok_s(nullptr, delchar, &p_tmp)));
  }
  delete[] buffer;
  return 0;
}

graphStatus DynamicGetNextCommonInfer(Operator &op) {
  std::vector<ge::DataType> output_types;
  if (op.GetAttr("output_types", output_types) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op get attr output_types failed");
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> output_shapes;
  if (op.GetAttr("output_shapes", output_shapes) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op get attr output_shapes failed");
    return GRAPH_FAILED;
  }

  std::string dynamic_graph_execute_mode;
  if (op.GetAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op get attr _dynamic_graph_execute_mode failed");
    return GRAPH_FAILED;
  }

  std::string getnext_inputs_shape_range;
  if (op.GetAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Op get attr _getnext_inputs_shape_range failed");
    return GRAPH_FAILED;
  }

  if (dynamic_graph_execute_mode != "lazy_recompile" && dynamic_graph_execute_mode != "dynamic_execute") {
    OP_LOGE(TbeGetName(op).c_str(), "Op get attr _dynamic_graph_execute_mode : [%s] is invailed",
            dynamic_graph_execute_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<std::vector<std::pair<int64_t, int64_t>>> shape_ranges;
  if (dynamic_graph_execute_mode == "dynamic_execute") {
    std::string regex_format = std::string("(\\[(((\\d+|(\\d+~\\d+)|-1),{1})*)(\\d+|(\\d+~\\d+)|-1)\\],{1})*") +
                               std::string("\\[(((\\d+|(\\d+~\\d+)|-1),{1})*)(\\d+|(\\d+~\\d+)|-1)\\]");
    std::regex r(regex_format, std::regex_constants::ECMAScript);
    if (!regex_match(getnext_inputs_shape_range, r)) {
      OP_LOGE(TbeGetName(op).c_str(),
              "The format of _getnext_inputs_shape_range is incorrect, and the parsing failed");
      return GRAPH_FAILED;
    }
    std::vector<std::string> shape_range_strings;
    int32_t res = Split(getnext_inputs_shape_range,
                        shape_range_strings, "]"); // shape_range_strings = [128,3~5,2~128,1],[128]
    if (res != 0) {
      OP_LOGE(TbeGetName(op).c_str(),
              "Split _getnext_inputs_shape_range failed");
      return GRAPH_FAILED;
    }

    for (auto shape_range_string : shape_range_strings) {
      if (shape_range_string.find("[") == 0) {
        shape_range_string.erase(0, 1);
      } else if (shape_range_string.find(",[") == 0) {
        shape_range_string.erase(0, 2);
      } else {
        OP_LOGE(TbeGetName(op).c_str(),
                "The format of _getnext_inputs_shape_range is incorrect, and the parsing failed");
        return GRAPH_FAILED;
      }
      // shape_range_string = 128,3~5,2~128,1
      std::vector<std::string> single_shape_ranges;
      std::vector<std::pair<int64_t, int64_t>> pair_shape_ranges;
      res = Split(shape_range_string, single_shape_ranges, ",");
      if (res != 0) {
        OP_LOGE(TbeGetName(op).c_str(),
                "Split _getnext_inputs_shape_range failed");
        return GRAPH_FAILED;
      }
      for (auto single_shape_range : single_shape_ranges) {
        std::vector<std::string> range;
        res = Split(single_shape_range, range, "~");
        if (res != 0) {
          OP_LOGE(TbeGetName(op).c_str(),
                  "Split _getnext_inputs_shape_range failed");
          return GRAPH_FAILED;
        }
        if (range.size() > kRangeMaxNum || range.size() <= 0) {
          OP_LOGE(TbeGetName(op).c_str(),
                  "The format of _getnext_inputs_shape_range is incorrect, and the parsing failed");
          return GRAPH_FAILED;
        }
        if (range.size() == kRangeMaxNum) {
          pair_shape_ranges.push_back(
              std::pair<int64_t, int64_t>{ atoi(range[0].c_str()), atoi(range[1].c_str()) });
        } else if (atoi(range[0].c_str()) == UNKNOWN_DIM) {
          pair_shape_ranges.push_back(
              std::pair<int64_t, int64_t>{1, -1});
        } else {
          pair_shape_ranges.push_back(
              std::pair<int64_t, int64_t>{ atoi(range[0].c_str()), atoi(range[0].c_str()) });
        }
      }
      shape_ranges.push_back(pair_shape_ranges);
    }

    if (shape_ranges.size() != output_shapes.size()) {
      OP_LOGE(TbeGetName(op).c_str(), "The _getnext_inputs_shape_range and output_shapes should be the same length.");
      return GRAPH_FAILED;
    }
    for (size_t i = 0; i < shape_ranges.size(); i++) {
      if (shape_ranges[i].size() != output_shapes[i].size()) {
        OP_LOGE(TbeGetName(op).c_str(),
                "The _getnext_inputs_shape_range and output_shapes at [%zu] should be the same length.", i);
        return GRAPH_FAILED;
      }
    }
  }

  if (output_types.size() != output_shapes.size()) {
    OP_LOGE(TbeGetName(op).c_str(), "The output_types and output_shapes should be the same length.");
    return GRAPH_FAILED;
  }

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  graphStatus output_status;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  for (int i = 0; i < static_cast<int>(output_shapes.size()); i++) {
    GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
    TensorDesc output_desc = op.GetDynamicOutputDesc("y", i);
    Shape shape(output_shapes[i]);
    output_desc.SetShape(Shape(output_shapes[i]));
    output_desc.SetDataType(output_types[i]);
    if (RankKnown(shape)) {
      auto dims = shape.GetDims();
      bool shape_fully_defined = true;
      for (const int64_t& dim : dims) {
        if (dim == UNKNOWN_DIM) {
          shape_fully_defined = false;
          break;
        }
      }
      if (!shape_fully_defined) {
        std::vector<std::pair<int64_t, int64_t>> shape_range;
        for (size_t j = 0; j < dims.size(); j++) {
          int64_t dim = dims[j];
          if (dynamic_graph_execute_mode == "lazy_recompile") {
            shape_range.push_back(dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} : \
                                                       std::pair<int64_t, int64_t>{dim, dim});
          } else {
            shape_range.push_back(shape_ranges[i][j]);
          }
        }
        output_desc.SetShapeRange(shape_range);
      }
    }
    output_status = op.UpdateDynamicOutputDesc("y", i, output_desc);
    if (output_status != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "Update [%s] [%d] failed", "y", i);
      return GRAPH_FAILED;
    }
  }
 
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DynamicGetNext, DynamicGetNextInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDescByName("x"), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Input x must be 0-D");
    return GRAPH_FAILED;
  }
  return DynamicGetNextCommonInfer(op);
}
INFER_FUNC_REG(DynamicGetNext, DynamicGetNextInfer);

IMPLEMT_INFERFUNC(DynamicGetNextV2, DynamicGetNextV2Infer) {
  return DynamicGetNextCommonInfer(op);
}
INFER_FUNC_REG(DynamicGetNextV2, DynamicGetNextV2Infer);

IMPLEMT_COMMON_INFERFUNC(FakeQueueInferShape) {
  OpDescPtr opDesc = OpDescUtils::GetOpDescFromOperator(op);
  if (opDesc == nullptr) {
      OP_LOGE(TbeGetName(op).c_str(), "Op desc is NULL!");
      return GRAPH_FAILED;
  }
  GeShape handleShape({2});
  auto handleDesc = opDesc->MutableOutputDesc("handle");
  (void)FillOpDesc(handleDesc, handleShape, DT_STRING);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FakeQueue, FakeQueueInferShape);

// --------------------------------LruCache-------------------------------------
IMPLEMT_COMMON_INFERFUNC(LruCacheInferShape) {
  int64_t cache_size;
  if (op.GetAttr("cache_size", cache_size) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr cache_size error.");
    return GRAPH_FAILED;
  }
  if (cache_size <= 0) {
    OP_LOGE(TbeGetName(op).c_str(), "cache_size should be >0 , real value is %ld.", cache_size);
    return GRAPH_PARAM_INVALID;
  }

  float load_factor;
  if (op.GetAttr("load_factor", load_factor) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr load_factor error.");
    return GRAPH_FAILED;
  }
  if (load_factor <= 0.0 || load_factor > 1.0) {
    OP_LOGE(TbeGetName(op).c_str(), "load_factor should be in  (0, 1.0] , real value is %f.", load_factor);
    return GRAPH_PARAM_INVALID;
  }
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDescByName("cache");
  DataType output_type = DT_RESOURCE;
  output_desc.SetShape(scalar_shape);
  output_desc.SetDataType(output_type);
  graphStatus output_status = op.UpdateOutputDesc("cache", output_desc);
  if (output_status != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update handle failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LruCache, LruCacheInferShape);
// ---------------------LruCache END-------------------------------------

// ----------------LRUCacheV2 Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(LRUCacheV2InferShape) {
  OP_LOGD("OP[LRUCacheV2]", "LRUCacheV2InferShape Begin.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto index_list_desc = op_info->MutableInputDesc(0);
  auto data_desc = op_info->MutableInputDesc(1);
  auto cache_desc = op_info->MutableInputDesc(2);
  auto tag_desc = op_info->MutableInputDesc(3);
  const GeShape& data_shape = data_desc->MutableShape();
  const GeShape& index_list_shape = index_list_desc->MutableShape();
  const GeShape& tag_shape = tag_desc->MutableShape();
  const GeShape& cache_shape = cache_desc->MutableShape();
  DataType index_list_dtype = index_list_desc->GetDataType();
  DataType data_dtype = data_desc->GetDataType();
  DataType tag_dtype = tag_desc->GetDataType();
  auto output_desc_data = op_info->MutableOutputDesc(0);
  auto output_desc_cache = op_info->MutableOutputDesc(1);
  auto output_desc_tag = op_info->MutableOutputDesc(2);
  auto output_desc_index_offset_list = op_info->MutableOutputDesc(3);
  auto output_desc_not_in_cache_index_list = op_info->MutableOutputDesc(4);
  auto output_desc_not_in_cache_number = op_info->MutableOutputDesc(5);
  output_desc_data->SetDataType(data_dtype);
  output_desc_cache->SetDataType(data_dtype);
  output_desc_tag->SetDataType(tag_dtype);
  output_desc_index_offset_list->SetDataType(index_list_dtype);
  output_desc_not_in_cache_index_list->SetDataType(index_list_dtype);
  output_desc_not_in_cache_number->SetDataType(index_list_dtype);
  output_desc_data->SetShape(data_shape);
  output_desc_cache->SetShape(cache_shape);
  output_desc_tag->SetShape(tag_shape);
  output_desc_index_offset_list->SetShape(index_list_shape);
  output_desc_not_in_cache_index_list->SetShape(index_list_shape);
  vector<int64_t> out_scalar_shape = {1};
  output_desc_not_in_cache_number->SetShape(GeShape(out_scalar_shape));
  if (IsUnknown(index_list_shape.GetDims())) {
    std::vector<std::pair<int64_t, int64_t>> input_shape_range;
    index_list_desc->GetShapeRange(input_shape_range);
    output_desc_index_offset_list->SetShapeRange(input_shape_range);
    output_desc_not_in_cache_index_list->SetShapeRange(input_shape_range);
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(LRUCacheV2, LRUCacheV2InferShape);
// ---------------- Op LRUCacheV2 End-------------------

// --------------------------------CacheAdd-------------------------------------
IMPLEMT_COMMON_INFERFUNC(CacheAddInferShape) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr ids = op_desc->MutableInputDesc(1);

  GeShape ids_shape;
  if (WithRank(ids, 1, ids_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call WithRank failed, ", GetShapeErrMsg(0,
            DebugString(ids->GetShape().GetDims()), "1D")));
    return GRAPH_FAILED;
  }

  GeTensorDescPtr swap_in_id_desc = op_desc->MutableOutputDesc(0);
  GeTensorDescPtr swap_in_idx_desc = op_desc->MutableOutputDesc(1);
  GeTensorDescPtr swap_out_id_desc = op_desc->MutableOutputDesc(2);
  GeTensorDescPtr swap_out_idx_desc = op_desc->MutableOutputDesc(3);
  swap_in_id_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_in_id_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_in_id_desc->SetDataType(ids->GetDataType());

  swap_in_idx_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_in_idx_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_in_idx_desc->SetDataType(ids->GetDataType());

  swap_out_id_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_out_id_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_out_id_desc->SetDataType(ids->GetDataType());

  swap_out_idx_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_out_idx_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_out_idx_desc->SetDataType(ids->GetDataType());

  if (ids_shape.GetShapeSize() != UNKNOWN_DIM) {
    std::vector<std::pair<int64_t, int64_t>> range;
    int64_t max_dim = ids_shape.GetDim(0);
    range.emplace_back(std::make_pair(1, max_dim));
    swap_in_id_desc->SetShapeRange(range);
    swap_in_idx_desc->SetShapeRange(range);
    swap_out_id_desc->SetShapeRange(range);
    swap_out_idx_desc->SetShapeRange(range);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CacheAdd, CacheAddInferShape);
// ---------------------CacheAdd END-------------------------------------

// --------------------------------CacheRemoteIndexToLocal--------------------------------
IMPLEMT_COMMON_INFERFUNC(CacheRemoteIndexToLocalInferShape) {
  TensorDesc desc = op.GetInputDesc(1);
  TensorDesc local_idx_desc = op.GetOutputDesc(0);
  Shape ids_shape = desc.GetShape();
  std::vector<std::pair<int64_t, int64_t>> range;
  auto status = desc.GetShapeRange(range);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  local_idx_desc.SetShapeRange(range);
  local_idx_desc.SetShape(ids_shape);
  local_idx_desc.SetDataType(desc.GetDataType());
  return op.UpdateOutputDesc("local_idx", local_idx_desc);
}

COMMON_INFER_FUNC_REG(CacheRemoteIndexToLocal, CacheRemoteIndexToLocalInferShape);
// ---------------------CacheRemoteIndexToLocal END-------------------------------------

// --------------------------------CacheAllIndexToLocal--------------------------------
IMPLEMT_COMMON_INFERFUNC(CacheAllIndexToLocalInferShape) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  TensorDesc local_idx_desc = op.GetOutputDesc(0);
  local_idx_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  local_idx_desc.SetDataType(dtype);
  return op.UpdateOutputDesc("local_idx", local_idx_desc);
}

COMMON_INFER_FUNC_REG(CacheAllIndexToLocal, CacheAllIndexToLocalInferShape);
// ---------------------CacheRemoteIndexToLocal END-------------------------------------

// ----------------------------AdpGetNext---------------------------------------
IMPLEMT_INFERFUNC(AdpGetNext, AdpGetNextInfer) {
  std::vector<ge::DataType> outputTypes;
  if (op.GetAttr("output_types", outputTypes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_types] failed"));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> outputShapes;
  if (op.GetAttr("output_shapes", outputShapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_shapes] failed"));
    return GRAPH_FAILED;
  }

  if (outputTypes.size() != outputShapes.size()) {
    std::string errMsg =
      "attr[output_types] and attr[output_shapes] should be the same length ";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < outputShapes.size(); i++) {
    TensorDesc outputDesc = op.GetDynamicOutputDesc("y", i);
    Shape shape(outputShapes[i]);
    outputDesc.SetShape(shape);
    outputDesc.SetDataType(outputTypes[i]);
    graphStatus status = op.UpdateDynamicOutputDesc("y", i, outputDesc);
    if (status != GRAPH_SUCCESS) {
      std::ostringstream sstr;
      sstr << "update output[y] index[";
      sstr << i;
      sstr << "] desc failed";
      std::string errMsg = sstr.str();
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdpGetNext, AdpGetNextInfer);
// ----------------------------AdpGetNext END-----------------------------------

// ----------------------------GetNextV2---------------------------------------
IMPLEMT_INFERFUNC(GetNextV2, GetNextV2Infer) {
  std::vector<ge::DataType> outputTypes;
  if (op.GetAttr("output_types", outputTypes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_types] failed"));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> outputShapes;
  if (op.GetAttr("output_shapes", outputShapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_shapes] failed"));
    return GRAPH_FAILED;
  }

  if (outputTypes.size() != outputShapes.size()) {
    std::string errMsg =
      "attr[output_types] and attr[output_shapes] should be the same length";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < outputShapes.size(); i++) {
    TensorDesc outputDesc = op.GetDynamicOutputDesc("y", i);
    Shape shape(outputShapes[i]);
    outputDesc.SetShape(shape);
    outputDesc.SetDataType(outputTypes[i]);
    graphStatus status = op.UpdateDynamicOutputDesc("y", i, outputDesc);
    if (status != GRAPH_SUCCESS) {
      std::ostringstream ss;
      ss << "update output[y] index[";
      ss << i;
      ss << "] desc failed";
      std::string errMsg = ss.str();
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GetNextV2, GetNextV2Infer);
// ----------------------------GetNextV2 END-----------------------------------

// ----------------------------GetNextFromQueue--------------------------------
IMPLEMT_INFERFUNC(GetNextFromQueue, GetNextFromQueueInferShape) {
  std::vector<ge::DataType> outputTypes;
  if (op.GetAttr("output_types", outputTypes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_types] failed"));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> outputShapes;
  if (op.GetAttr("output_shapes", outputShapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[output_shapes] failed"));
    return GRAPH_FAILED;
  }

  if (outputTypes.size() != outputShapes.size()) {
    std::string errMsg =
      "attr[output_types] and attr[output_shapes] should be the same length";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < outputShapes.size(); i++) {
    TensorDesc outputDesc = op.GetDynamicOutputDesc("y", i);
    Shape shape(outputShapes[i]);
    outputDesc.SetShape(shape);
    outputDesc.SetDataType(outputTypes[i]);
    graphStatus outputStatus = op.UpdateDynamicOutputDesc("y", i, outputDesc);
    if (outputStatus != GRAPH_SUCCESS) {
      std::ostringstream ss;
      ss << "update output[y] index[";
      ss << i;
      ss << "] desc failed";
      std::string errMsg = ss.str();
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), errMsg);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(GetNextFromQueue, GetNextFromQueueInferShape);
// ----------------------------GetNextFromQueue END----------------------------

// ----------------------------OptionalGetValue--------------------------------
IMPLEMT_INFERFUNC(OptionalGetValue, OptionalGetValueInferShape) {
  OP_LOGD("OP[OptionalGetValue]", "OptionalGetValueInferShape Begin.");
  const auto op_name = TbeGetName(op);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto optional_ptr = op_desc->MutableInputDesc("optional");
  GeShape optional_shape;
  if (WithRank(optional_ptr, 0, optional_shape, op_name.c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Expected optional should be 0-D. But get %lu.", optional_ptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  std::vector<ge::DataType> outputTypes;
  if (op.GetAttr("output_types", outputTypes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, string("get attr[output_types] failed"));
    return GRAPH_FAILED;
  }

  size_t typesSize = outputTypes.size();
  if (typesSize < 1) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, string("get attr[output_types] less 1"));
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> outputShapes;
  if (op.GetAttr("output_shapes", outputShapes) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, string("get attr[output_shapes] failed"));
    return GRAPH_FAILED;
  }

  size_t shapesSize = outputShapes.size();
  if (shapesSize < 1) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, string("get attr[outputShapes] less 1"));
    return GRAPH_FAILED;
  }

  if (typesSize != shapesSize) {
    std::string errMsg =
      "attr[output_types] and attr[output_shapes] should be the same length";
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, errMsg);
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < shapesSize; i++) {
    TensorDesc outputDesc = op.GetDynamicOutputDesc("components", i);
    Shape shape(outputShapes[i]);
    outputDesc.SetShape(shape);
    outputDesc.SetDataType(outputTypes[i]);
  
    graphStatus outputStatus = op.UpdateDynamicOutputDesc("components", i, outputDesc);
    if (outputStatus != GRAPH_SUCCESS) {
      std::ostringstream sInfo;
      sInfo << "update output[components] index[";
      sInfo << i;
      sInfo << "] desc failed";
      std::string errMsg = sInfo.str();
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, errMsg);
      return GRAPH_FAILED;
    }
  }

  OP_LOGD("OP[OptionalGetValue]", "OptionalGetValueInferShape End.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OptionalGetValue, OptionalGetValueInferShape);
// ----------------------------OptionalGetValue END----------------------------
}  // namespace ge
