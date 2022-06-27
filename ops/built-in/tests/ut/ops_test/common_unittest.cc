/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "common/utils/ut_op_common.h"

void CommonInferShapeOperatorWithConst(ge::Operator& op, vector<bool> input_const, vector<string> attrs,
                                       vector<vector<int64_t>> expect_shapes) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_CONST(holder, op, input_const, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorWithConstFail(ge::Operator& op, vector<bool> input_const,
                                           vector<string> attrs) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_CONST(holder, op, input_const, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_FAILED);
}

void CommonInferShapeOperatorWithIrNum(ge::Operator& op, vector<uint32_t> irnum,
                                       vector<string> attrs, vector<vector<int64_t>> expect_shapes) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_IRNUM(holder, op, irnum, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorWithIrNumFail(ge::Operator& op, vector<uint32_t> irnum,
                                           vector<string> attrs) {
  ATTACH_OPERATOR_TO_HOLDER_WITH_IRNUM(holder, op, irnum, attrs);

  std::string optype = op.GetOpType();
  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_FAILED);
}

void CommonInferShapeOperator(ge::Operator& op, vector<string> attrs, std::vector<std::vector<int64_t>> expect_shapes) {
  std::string optype = op.GetOpType();

  ATTACH_OPERATOR_TO_HOLDER(holder, op, attrs);

  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_SUCCESS);

  size_t expect_count = std::min(output_size, expect_shapes.size());
  for (size_t i = 0; i < expect_count; i++) {
    VERIFY_OUTPUT_SHAPE(holder, i, expect_shapes[i]);
  }
}

void CommonInferShapeOperatorFail(ge::Operator& op, vector<string> attrs) {
  std::string optype = op.GetOpType();

  ATTACH_OPERATOR_TO_HOLDER(holder, op, attrs);

  HOLDER_DO_INFER_SHAPE(holder, optype, GRAPH_FAILED);
}

ge::graphStatus InferShapeTest(ge::Operator& op, const Runtime2TestParam& param) {
  ge::graphStatus ret;
  size_t input_size = op.GetInputsSize();
  std::vector<std::string> attrs = param.attrs;
  std::vector<bool> input_const = param.input_const;
  std::vector<uint32_t> irnum = param.irnum;
  if (irnum.size() > 0) {
    if (input_const.size() == 0) input_const.assign(irnum.size(), false);
  } else if (input_const.size() > 0) {
    if (irnum.size() == 0) irnum.assign(input_const.size(), 1);
  } else {
    input_const.assign(input_size, false);
    irnum.assign(input_size, 1);
  }
  size_t output_size = op.GetOutputsSize();
  auto faker = gert::InferShapeContextFaker()
                    .NodeIoNum(input_size, output_size)
                    .IrInstanceNum(irnum);

  vector<uint8_t*> const_tensors;
  std::vector<gert::StorageShape> input_shapes(input_size);
  std::vector<void *> input_shapes_ref(input_size);
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);
  if (operator_info == nullptr) return GRAPH_FAILED;
  if (input_size > 0) {
    size_t count = 0;
    for (size_t i = 0; i < input_const.size(); ++i) {
      if (input_const[i]) {
        uint8_t* input_tensor_holder = GetConstTensor(op, i);
        if (input_tensor_holder == nullptr) return GRAPH_FAILED;
        input_shapes_ref[count] = input_tensor_holder;
        const_tensors.push_back(input_tensor_holder);
        count++;
      } else for (int idx = 0; idx < irnum[i]; idx++) {
        auto input_desc = operator_info->MutableInputDesc(i + idx);
        if (input_desc == nullptr) continue;
        ge::Format input_format = input_desc->GetFormat();
        ge::Format origin_format = input_desc->GetOriginFormat();
        ge::DataType dtype = input_desc->GetDataType();
        faker = faker.NodeInputTd(count, dtype, origin_format, input_format);
        for (int64_t dim : input_desc->GetOriginShape().GetDims()) {
          input_shapes[count].MutableOriginShape().AppendDim(dim);
        }
        for (int64_t dim : input_desc->MutableShape().GetDims()) {
          input_shapes[count].MutableStorageShape().AppendDim(dim);
        }
        input_shapes_ref[count] = &input_shapes[count];
        count++;
      }
    }
    faker = faker.InputShapes(input_shapes_ref);
  }

  std::vector<gert::StorageShape> output_shapes(output_size);
  std::vector<void *> output_shapes_ref(output_size);
  if (output_size > 0) {
    for (size_t i = 0; i < output_size; ++i) {
      output_shapes_ref[i] = &output_shapes[i];
    }
    faker = faker.OutputShapes(output_shapes_ref);
  }

  auto op_attrs_map = op.GetAllAttrNamesAndTypes();
  std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value;
  std::pair<std::string, ge::AnyValue> p;
  if (attrs.size() > 0) {
    for (auto item : attrs) {
      p.first = item;
      auto attr_it = op_attrs_map.find(item);
      if (attr_it != op_attrs_map.end()) {
        auto type_it = kAttrTypesMap.find(attr_it->second);
        if (type_it != kAttrTypesMap.end()) {
          switch (type_it->second) {
            case ge::AnyValue::ValueType::VT_BOOL: {
              bool value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<bool>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_INT: {
              int64_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<int64_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_FLOAT: {
              float32_t value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<float32_t>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_STRING: {
              std::string value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::string>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_INT: {
              std::vector<int64_t> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<int64_t>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_BOOL: {
              std::vector<bool> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<bool>>(value);
            }
            break;
            case ge::AnyValue::ValueType::VT_LIST_LIST_INT: {
              std::vector<std::vector<int64_t>> value;
              if(op.GetAttr(item, value) != GRAPH_SUCCESS) return GRAPH_FAILED;
              p.second = ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(value);
            }
            break;
          }
        }
      }
      keys_to_value.push_back(p);
    }
    faker = faker.NodeAttrs(keys_to_value);
  }

  auto holder = faker.Build();
  std::string optype = op.GetOpType();
  if (gert::OpImplRegistry::GetInstance().GetOpImpl(optype) == nullptr) return GRAPH_FAILED;
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->infer_shape;
  if (infer_shape_func == nullptr) return GRAPH_FAILED;
  gert::InferShapeContext *context = holder.GetContext<gert::InferShapeContext>();
  if (context == nullptr) return GRAPH_FAILED;
  ret = infer_shape_func(context);
  for (uint8_t* tensor : const_tensors) { delete []tensor; }
  for (size_t i = 0; i < output_size; i++) {
    auto out_shape = context->GetOutputShape(i);
    if (out_shape == nullptr) return GRAPH_FAILED;
    auto output_desc = operator_info->MutableOutputDesc(i);
    output_desc->SetShape(GeShape(ops::ToVector(*out_shape)));
  }
  return ret;
}

ge::graphStatus InferShapeTest(ge::Operator& op) {
  Runtime2TestParam param;
  return InferShapeTest(op, param);
}
