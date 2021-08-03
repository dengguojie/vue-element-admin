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
 * \file index_to_addr_ops.cpp
 * \brief
 */
#include "inc/index_to_addr_ops.h"

#include <vector>

#include "op_log.h"
#include "util/error_util.h"
#include "util/util.h"

namespace ge {
IMPLEMT_INFERFUNC(IndexToAddr, IndexToAddrInferShape) {
  TensorDesc base_addr_desc = op.GetInputDescByName("base_addr");
  DataType base_addr_dtype = base_addr_desc.GetDataType();
  Format base_addr_format = base_addr_desc.GetFormat();

  AscendString opName;
  CHECK(op.GetName(opName) != GRAPH_SUCCESS, OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  std::vector<int64_t> block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        opName.GetString(), string("failed to get attr[block_size]."));
    return GRAPH_FAILED;
  }

  if (block_size.size() != 2) {
    string error_msg = ConcatString("Attr block_size size[", block_size.size(),
                                    "] must be 2.");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(opName.GetString(), error_msg);
    return GRAPH_FAILED;
  }

  TensorDesc addrs_table_desc = op.GetOutputDescByName("addrs_table");
  addrs_table_desc.SetFormat(base_addr_format);
  addrs_table_desc.SetShape(Shape({block_size[0], 4}));
  addrs_table_desc.SetDataType(base_addr_dtype);
  if (op.UpdateOutputDesc("addrs_table", addrs_table_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        opName.GetString(), string("update description for [addrs_table] failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IndexToAddr, IndexToAddrInferShape);
}  // namespace ge
