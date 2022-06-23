/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include <iostream>
#include <string.h>
#include "ge_ir_build.h"
#include "all_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;


bool GenGraph(Graph& graph)
{
    const int32_t KN_FP16_VALUE = 16;   // k0,n0 value in fp16
    const int32_t DYNAMIC_VALUE = -1;   // -1 means support any value
    const int32_t DYNAMIC_ALL_VALUE = -2;  // -2 means support any shape
    // since transdata default generalization
    vector<int64_t> dim_transdata1_in({DYNAMIC_ALL_VALUE});
    ge::Shape shape_transdata1_in(dim_transdata1_in);
    vector<std::pair<int64_t, int64_t>> range_transdata1_in =
        {{1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}};
    TensorDesc desc_transdata1_in(shape_transdata1_in);
    desc_transdata1_in.SetOriginFormat(FORMAT_NCHW);
    desc_transdata1_in.SetOriginShape(shape_transdata1_in);
    desc_transdata1_in.SetFormat(FORMAT_NCHW);
    desc_transdata1_in.SetDataType(DT_FLOAT16);
    desc_transdata1_in.SetShapeRange(range_transdata1_in);

    // data op
    auto data = op::Data("data");
    data.update_input_desc_x(desc_transdata1_in);
    data.update_output_desc_y(desc_transdata1_in);

    vector<int64_t> dim_transdata1_out({DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, KN_FP16_VALUE});
    ge::Shape shape_transdata1_out(dim_transdata1_out);
    vector<std::pair<int64_t, int64_t>> range_transdata1_out =
        {{1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE},
         {1, DYNAMIC_VALUE}, {KN_FP16_VALUE, KN_FP16_VALUE}};
    TensorDesc desc_transdata1_out(shape_transdata1_out);
    desc_transdata1_out.SetFormat(FORMAT_NC1HWC0);
    desc_transdata1_out.SetDataType(DT_FLOAT16);
    desc_transdata1_out.SetShapeRange(range_transdata1_out);
    desc_transdata1_out.SetOriginShape(shape_transdata1_in);

    // transdata 1
    auto transdata_pre = op::TransData("transdata1")
        .set_input_src(data)
        .set_attr_src_format("NCHW")
        .set_attr_dst_format("NC1HWC0")
        .set_attr_groups(DYNAMIC_VALUE);

    transdata_pre.SetAttr("var_attrs", "groups");

    transdata_pre.update_input_desc_src(desc_transdata1_in);
    transdata_pre.update_output_desc_dst(desc_transdata1_out);
    transdata_pre.SetAttr("graph_pattern", "_transdata_conv2d_transdata");

    vector<int64_t> dim_filter({DYNAMIC_VALUE, DYNAMIC_VALUE, KN_FP16_VALUE, KN_FP16_VALUE});
    ge::Shape shape_filter(dim_filter);
    vector<int64_t> dim_filter_ori({DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE});
    ge::Shape shape_filter_ori(dim_filter_ori);
    TensorDesc filter_desc(shape_filter);
    filter_desc.SetFormat(FORMAT_FRACTAL_Z);
    filter_desc.SetDataType(DT_FLOAT16);
    filter_desc.SetOriginShape(shape_filter_ori);
    filter_desc.SetOriginFormat(FORMAT_NCHW);

    // filter data op
    auto filter_data = op::Data("filter_data");
    filter_data.update_input_desc_x(filter_desc);
    filter_data.update_output_desc_y(filter_desc);

    vector<int64_t> dim_transdata2_in({DYNAMIC_ALL_VALUE});
    ge::Shape shape_transdata2_in(dim_transdata2_in);
    vector<std::pair<int64_t, int64_t>> range_transdata2_in =
        {{1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE},
         {1, DYNAMIC_VALUE}, {KN_FP16_VALUE, KN_FP16_VALUE}};
    TensorDesc desc_transdata2_in(shape_transdata2_in);
    desc_transdata2_in.SetFormat(FORMAT_NC1HWC0);
    desc_transdata2_in.SetDataType(DT_FLOAT16);
    desc_transdata2_in.SetShapeRange(range_transdata2_in);

    // conv2d op
    auto conv2d = op::Conv2D("Conv2d")
        .set_input_x(transdata_pre)
        .set_input_filter(filter_data)
        .set_attr_strides({DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE})
        .set_attr_pads({DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE})
        .set_attr_dilations({DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE, DYNAMIC_VALUE})
        .set_attr_groups(DYNAMIC_VALUE)
        .set_attr_data_format("NCHW");
    conv2d.update_input_desc_x(desc_transdata1_out);
    conv2d.update_input_desc_filter(filter_desc);
    conv2d.update_output_desc_y(desc_transdata2_in);

    conv2d.SetAttr("pattern", "Convolution");
    vector<AscendString> conv_attrs({"strides", "pads", "dilations", "groups", "offset_x"});
    conv2d.SetAttr("var_attrs", conv_attrs);

    vector<int64_t> dim_transdata2_out({DYNAMIC_ALL_VALUE});
    ge::Shape shape_transdata2_out(dim_transdata2_out);
    vector<std::pair<int64_t, int64_t>> range_transdata2_out =
        {{1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}, {1, DYNAMIC_VALUE}};
    TensorDesc desc_transdata2_out(shape_transdata2_out);
    desc_transdata2_out.SetFormat(FORMAT_NCHW);
    desc_transdata2_out.SetDataType(DT_FLOAT16);
    desc_transdata2_out.SetShapeRange(range_transdata2_out);
    desc_transdata2_out.SetOriginFormat(FORMAT_NCHW);

    // transdata 2
    auto transdata_post = op::TransData("transdata2")
        .set_input_src(conv2d)
        .set_attr_src_format("NC1HWC0")
        .set_attr_dst_format("NCHW")
        .set_attr_groups(DYNAMIC_VALUE);

    transdata_post.SetAttr("var_attrs", "groups");

    transdata_post.update_input_desc_src(desc_transdata2_in);
    transdata_post.update_output_desc_dst(desc_transdata2_out);

    std::vector<Operator> inputs{data, filter_data};
    std::vector<Operator> outputs{transdata_post};
    graph.SetInputs(inputs).SetOutputs(outputs);

    return true;
}

int main(int argc, char* argv[])
{
    // Generate graph
    Graph graph1("IrGraph1");
    bool ret = GenGraph(graph1);
    if (!ret) {
        cout << "=========== Generate Graph1 Failed! ===========" << endl;
        return -1;
    }
    cout << "=========== Generate Graph1 Success! ===========" << endl;
    const char *file_name = "conv2d_net";
    bool graphStatus = aclgrphDumpGraph(graph1, file_name, strlen(file_name));
    if (graphStatus != GRAPH_SUCCESS) {
        cout << "=========== Dump Graph1 Failed! ===========" << endl;
        return -1;
    }
    cout << "=========== Dump Graph1 Success! ===========" << endl;
    return 0;
}