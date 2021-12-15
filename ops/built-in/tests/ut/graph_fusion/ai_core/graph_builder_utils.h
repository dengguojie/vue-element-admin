/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef GRAPH_FUSION_GRAPH_BUILDER_UTILS_H_
#define GRAPH_FUSION_GRAPH_BUILDER_UTILS_H_

#include <string>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"

namespace ut {
using namespace ge;

struct UtOpDesc {
  Format format;
  DataType dtype;
  std::vector<int64_t> shape;

  GeTensorDesc GetGeTensorDesc() const {
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape(shape));
    tensor_desc.SetFormat(format);
    tensor_desc.SetDataType(dtype);
    tensor_desc.SetOriginFormat(format);
    tensor_desc.SetOriginShape(GeShape(shape));
    tensor_desc.SetOriginDataType(dtype);
    return tensor_desc;
  }
};

class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  const std::vector<int64_t>& shape = {8, 1, 224, 224},
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT);
  NodePtr AddNode(const std::string &name, const std::string &type,
                  std::vector<std::string> input_names,
                  std::vector<std::string> output_names,
                  const std::vector<int64_t>& shape,
                  Format format, DataType data_type);
  NodePtr AddNode(const std::string &name,
                  const std::string &type,
                  const std::vector<UtOpDesc> &inputs,
                  const std::vector<UtOpDesc> &outputs);
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace ut

#endif  // GRAPH_FUSION_GRAPH_BUILDER_UTILS_H_
