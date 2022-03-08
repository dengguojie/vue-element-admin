#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;
namespace fe{
class MultiHeadAttentionFusionTest : public testing::Test {
protected:
    static void SetUpTestCase() {
    std::cout << "multi_head_attention_fusion_test SetUp" << std::endl;
}

    static void TearDownTestCase() {
    std::cout << "multi_head_attention_fusion_test TearDown" << std::endl;
}
};

static void set_nz_tensor_desc(ge::TensorDesc &tensor_desc, const vector<int64_t> &ori_dims, const ge::DataType dtype = DT_FLOAT16) {
    vector<int64_t> dims;
    int32_t dim = ori_dims.size();
    for (auto i = 0; i < dim - 2; i++) {
        dims.push_back(ori_dims[i]);
    }
    dims.push_back(ori_dims[dim-1]/16);
    dims.push_back(ori_dims[dim-2]/16);
    dims.push_back(16);
    dims.push_back(16);
    tensor_desc.SetShape(ge::Shape(dims));
    tensor_desc.SetDataType(dtype);
    tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
    tensor_desc.SetOriginShape(ge::Shape(ori_dims));
    tensor_desc.SetOriginFormat(FORMAT_ND);
}

static void set_nd_tensor_desc(ge::TensorDesc &tensorDesc, const vector<int64_t> &ori_dims, const ge::DataType dtype = DT_FLOAT16) {
    tensorDesc.SetShape(ge::Shape(ori_dims));
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(FORMAT_ND);
    tensorDesc.SetOriginShape(ge::Shape(ori_dims));
    tensorDesc.SetOriginFormat(FORMAT_ND);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_1) {
    ge::Graph graph("multi_head_attention_fusion_test_1");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch * tgt_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> kv_shape = {batch * src_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num * attn_dim_per_head, attn_head_num * attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num * attn_dim_per_head};
    vector<int64_t> attn_mask_shape = {batch, attn_head_num, tgt_len, src_len};
    vector<int64_t> dropout_mask_shape = {batch * attn_head_num * tgt_len * src_len / 8};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, attn_mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto out_proj_bias_data = op::Data("out_proj_bias");
    ge::TensorDesc out_proj_bias_tensor_desc;
    set_nz_tensor_desc(out_proj_bias_tensor_desc, bias_shape);
    out_proj_bias_data.update_input_desc_x(out_proj_bias_tensor_desc);
    out_proj_bias_data.update_output_desc_y(out_proj_bias_tensor_desc);

    auto dropout_mask_data = op::Data("dropout_mask");
    ge::TensorDesc dropout_mask_tensor_desc;
    set_nd_tensor_desc(dropout_mask_tensor_desc, dropout_mask_shape, DT_UINT8);
    dropout_mask_data.update_input_desc_x(dropout_mask_tensor_desc);
    dropout_mask_data.update_output_desc_y(dropout_mask_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_input_out_proj_bias(out_proj_bias_data) \
        .set_input_dropout_mask(dropout_mask_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs = {query_data, key_data, value_data, query_weight_data, key_weight_data,
        value_weight_data, attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, 
        value_bias_data, out_proj_bias_data, dropout_mask_data
    };
    std::vector<Operator> outputs = {MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_2) {
    ge::Graph graph("multi_head_attention_fusion_test_2");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch*tgt_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> kv_shape = {batch*src_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num*attn_dim_per_head, attn_head_num*attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num*attn_dim_per_head};
    vector<int64_t> mask_shape = {batch, attn_head_num, tgt_len, src_len};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto out_proj_bias_data = op::Data("out_proj_bias");
    ge::TensorDesc out_proj_bias_tensor_desc;
    set_nz_tensor_desc(out_proj_bias_tensor_desc, bias_shape);
    out_proj_bias_data.update_input_desc_x(out_proj_bias_tensor_desc);
    out_proj_bias_data.update_output_desc_y(out_proj_bias_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_input_out_proj_bias(out_proj_bias_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs{query_data, key_data, value_data, query_weight_data, key_weight_data, value_weight_data,
        attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, value_bias_data, out_proj_bias_data};
    std::vector<Operator> outputs{MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_3) {
    ge::Graph graph("multi_head_attention_fusion_test_3");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch*tgt_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> kv_shape = {batch*src_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num*attn_dim_per_head, attn_head_num*attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num*attn_dim_per_head};
    vector<int64_t> mask_shape = {batch, attn_head_num, tgt_len, src_len};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs{query_data, key_data, value_data, query_weight_data, key_weight_data, value_weight_data,
        attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, value_bias_data};
    std::vector<Operator> outputs{MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_4) {
    ge::Graph graph("multi_head_attention_fusion_test_4");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch*tgt_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> kv_shape = {batch*src_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num*attn_dim_per_head, attn_head_num*attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num*attn_dim_per_head};
    vector<int64_t> mask_shape = {batch, attn_head_num, tgt_len, src_len};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs{query_data, key_data, value_data, query_weight_data, key_weight_data, value_weight_data,
        attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data};
    std::vector<Operator> outputs{MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_5) {
    ge::Graph graph("multi_head_attention_fusion_test_5");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch*tgt_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> kv_shape = {batch*src_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num*attn_dim_per_head, attn_head_num*attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num*attn_dim_per_head};
    vector<int64_t> mask_shape = {batch, attn_head_num, tgt_len, src_len};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs{query_data, key_data, value_data, query_weight_data, key_weight_data, value_weight_data,
        attn_mask_data, out_proj_weight_data, query_bias_data};
    std::vector<Operator> outputs{MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_6) {
    ge::Graph graph("multi_head_attention_fusion_test_6");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch*tgt_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> kv_shape = {batch*src_len, attn_head_num*attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num*attn_dim_per_head, attn_head_num*attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num*attn_dim_per_head};
    vector<int64_t> mask_shape = {batch, attn_head_num, tgt_len, src_len};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs{query_data, key_data, value_data, query_weight_data, key_weight_data, value_weight_data,
        attn_mask_data, out_proj_weight_data};
    std::vector<Operator> outputs{MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_7) {
    ge::Graph graph("multi_head_attention_fusion_test_7");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 0.9;
    bool softmax_use_float = false;

    vector<int64_t> query_shape = {batch * tgt_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> kv_shape = {batch * src_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num * attn_dim_per_head, attn_head_num * attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num * attn_dim_per_head};
    vector<int64_t> attn_mask_shape = {batch, attn_head_num, tgt_len, src_len};
    vector<int64_t> dropout_mask_shape = {batch * attn_head_num * tgt_len * src_len / 8};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, attn_mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto out_proj_bias_data = op::Data("out_proj_bias");
    ge::TensorDesc out_proj_bias_tensor_desc;
    set_nz_tensor_desc(out_proj_bias_tensor_desc, bias_shape);
    out_proj_bias_data.update_input_desc_x(out_proj_bias_tensor_desc);
    out_proj_bias_data.update_output_desc_y(out_proj_bias_tensor_desc);

    auto dropout_mask_data = op::Data("dropout_mask");
    ge::TensorDesc dropout_mask_tensor_desc;
    set_nz_tensor_desc(dropout_mask_tensor_desc, dropout_mask_shape);
    dropout_mask_data.update_input_desc_x(dropout_mask_tensor_desc);
    dropout_mask_data.update_output_desc_y(dropout_mask_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_input_out_proj_bias(out_proj_bias_data) \
        .set_input_dropout_mask(dropout_mask_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs = {query_data, key_data, value_data, query_weight_data, key_weight_data,
        value_weight_data, attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, 
        value_bias_data, out_proj_bias_data, dropout_mask_data
    };
    std::vector<Operator> outputs = {MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_8) {
    ge::Graph graph("multi_head_attention_fusion_test_8");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 16;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 1.0;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch * tgt_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> kv_shape = {batch * src_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num * attn_dim_per_head, attn_head_num * attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num * attn_dim_per_head};
    vector<int64_t> attn_mask_shape = {batch, attn_head_num, tgt_len, src_len};
    vector<int64_t> dropout_mask_shape = {batch * attn_head_num * tgt_len * src_len / 8};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, attn_mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto out_proj_bias_data = op::Data("out_proj_bias");
    ge::TensorDesc out_proj_bias_tensor_desc;
    set_nz_tensor_desc(out_proj_bias_tensor_desc, bias_shape);
    out_proj_bias_data.update_input_desc_x(out_proj_bias_tensor_desc);
    out_proj_bias_data.update_output_desc_y(out_proj_bias_tensor_desc);

    auto dropout_mask_data = op::Data("dropout_mask");
    ge::TensorDesc dropout_mask_tensor_desc;
    set_nz_tensor_desc(dropout_mask_tensor_desc, dropout_mask_shape);
    dropout_mask_data.update_input_desc_x(dropout_mask_tensor_desc);
    dropout_mask_data.update_output_desc_y(dropout_mask_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_input_out_proj_bias(out_proj_bias_data) \
        .set_input_dropout_mask(dropout_mask_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs = {query_data, key_data, value_data, query_weight_data, key_weight_data,
        value_weight_data, attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, 
        value_bias_data, out_proj_bias_data, dropout_mask_data
    };
    std::vector<Operator> outputs = {MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, false);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_9) {
    ge::Graph graph("multi_head_attention_fusion_test_9");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = -1;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 1.0;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch * tgt_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> kv_shape = {batch * src_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num * attn_dim_per_head, attn_head_num * attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num * attn_dim_per_head};
    vector<int64_t> attn_mask_shape = {batch, attn_head_num, tgt_len, src_len};
    vector<int64_t> dropout_mask_shape = {batch * attn_head_num * tgt_len * src_len / 8};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, attn_mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto out_proj_bias_data = op::Data("out_proj_bias");
    ge::TensorDesc out_proj_bias_tensor_desc;
    set_nz_tensor_desc(out_proj_bias_tensor_desc, bias_shape);
    out_proj_bias_data.update_input_desc_x(out_proj_bias_tensor_desc);
    out_proj_bias_data.update_output_desc_y(out_proj_bias_tensor_desc);

    auto dropout_mask_data = op::Data("dropout_mask");
    ge::TensorDesc dropout_mask_tensor_desc;
    set_nz_tensor_desc(dropout_mask_tensor_desc, dropout_mask_shape);
    dropout_mask_data.update_input_desc_x(dropout_mask_tensor_desc);
    dropout_mask_data.update_output_desc_y(dropout_mask_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_input_out_proj_bias(out_proj_bias_data) \
        .set_input_dropout_mask(dropout_mask_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs = {query_data, key_data, value_data, query_weight_data, key_weight_data,
        value_weight_data, attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, 
        value_bias_data, out_proj_bias_data, dropout_mask_data
    };
    std::vector<Operator> outputs = {MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, true);
}

TEST_F(MultiHeadAttentionFusionTest, multi_head_attention_fusion_test_10) {
    ge::Graph graph("multi_head_attention_fusion_test_10");
    int64_t batch, attn_head_num, attn_dim_per_head, src_len, tgt_len;
    batch = 8;
    attn_head_num = 17;
    attn_dim_per_head = 64;
    src_len = 64;
    tgt_len = 64;
    float keep_prob = 1.0;
    bool softmax_use_float = true;

    vector<int64_t> query_shape = {batch * tgt_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> kv_shape = {batch * src_len, attn_head_num * attn_dim_per_head};
    vector<int64_t> weight_shape = {attn_head_num * attn_dim_per_head, attn_head_num * attn_dim_per_head};
    vector<int64_t> bias_shape = {attn_head_num * attn_dim_per_head};
    vector<int64_t> attn_mask_shape = {batch, attn_head_num, tgt_len, src_len};
    vector<int64_t> dropout_mask_shape = {batch * attn_head_num * tgt_len * src_len / 8};

    auto query_data = op::Data("query");
    ge::TensorDesc query_tensor_desc;
    set_nz_tensor_desc(query_tensor_desc, query_shape);
    query_data.update_input_desc_x(query_tensor_desc);
    query_data.update_output_desc_y(query_tensor_desc);

    auto key_data = op::Data("key");
    ge::TensorDesc key_tensor_desc;
    set_nz_tensor_desc(key_tensor_desc, kv_shape);
    key_data.update_input_desc_x(key_tensor_desc);
    key_data.update_output_desc_y(key_tensor_desc);

    auto value_data = op::Data("value");
    ge::TensorDesc value_tensor_desc;
    set_nz_tensor_desc(value_tensor_desc, kv_shape);
    value_data.update_input_desc_x(value_tensor_desc);
    value_data.update_output_desc_y(value_tensor_desc);

    auto query_weight_data = op::Data("query_weight");
    ge::TensorDesc query_weight_tensor_desc;
    set_nz_tensor_desc(query_weight_tensor_desc, weight_shape);
    query_weight_data.update_input_desc_x(query_weight_tensor_desc);
    query_weight_data.update_output_desc_y(query_weight_tensor_desc);

    auto key_weight_data = op::Data("key_weight");
    ge::TensorDesc key_weight_tensor_desc;
    set_nz_tensor_desc(key_weight_tensor_desc, weight_shape);
    key_weight_data.update_input_desc_x(key_weight_tensor_desc);
    key_weight_data.update_output_desc_y(key_weight_tensor_desc);

    auto value_weight_data = op::Data("value_weight");
    ge::TensorDesc value_weight_tensor_desc;
    set_nz_tensor_desc(value_weight_tensor_desc, weight_shape);
    value_weight_data.update_input_desc_x(value_weight_tensor_desc);
    value_weight_data.update_output_desc_y(value_weight_tensor_desc);

    auto attn_mask_data = op::Data("attn_mask");
    ge::TensorDesc attn_mask_tensor_desc;
    set_nz_tensor_desc(attn_mask_tensor_desc, attn_mask_shape);
    attn_mask_data.update_input_desc_x(attn_mask_tensor_desc);
    attn_mask_data.update_output_desc_y(attn_mask_tensor_desc);

    auto out_proj_weight_data = op::Data("out_proj_weight");
    ge::TensorDesc out_proj_weight_tensor_desc;
    set_nz_tensor_desc(out_proj_weight_tensor_desc, weight_shape);
    out_proj_weight_data.update_input_desc_x(out_proj_weight_tensor_desc);
    out_proj_weight_data.update_output_desc_y(out_proj_weight_tensor_desc);

    auto query_bias_data = op::Data("query_bias");
    ge::TensorDesc query_bias_tensor_desc;
    set_nz_tensor_desc(query_bias_tensor_desc, bias_shape);
    query_bias_data.update_input_desc_x(query_bias_tensor_desc);
    query_bias_data.update_output_desc_y(query_bias_tensor_desc);

    auto key_bias_data = op::Data("key_bias");
    ge::TensorDesc key_bias_tensor_desc;
    set_nz_tensor_desc(key_bias_tensor_desc, bias_shape);
    key_bias_data.update_input_desc_x(key_bias_tensor_desc);
    key_bias_data.update_output_desc_y(key_bias_tensor_desc);

    auto value_bias_data = op::Data("value_bias");
    ge::TensorDesc value_bias_tensor_desc;
    set_nz_tensor_desc(value_bias_tensor_desc, bias_shape);
    value_bias_data.update_input_desc_x(value_bias_tensor_desc);
    value_bias_data.update_output_desc_y(value_bias_tensor_desc);

    auto out_proj_bias_data = op::Data("out_proj_bias");
    ge::TensorDesc out_proj_bias_tensor_desc;
    set_nz_tensor_desc(out_proj_bias_tensor_desc, bias_shape);
    out_proj_bias_data.update_input_desc_x(out_proj_bias_tensor_desc);
    out_proj_bias_data.update_output_desc_y(out_proj_bias_tensor_desc);

    auto dropout_mask_data = op::Data("dropout_mask");
    ge::TensorDesc dropout_mask_tensor_desc;
    set_nz_tensor_desc(dropout_mask_tensor_desc, dropout_mask_shape);
    dropout_mask_data.update_input_desc_x(dropout_mask_tensor_desc);
    dropout_mask_data.update_output_desc_y(dropout_mask_tensor_desc);

    auto MultiHeadAttentionOp = op::MultiHeadAttention("multi_head_attention");
    MultiHeadAttentionOp.set_input_query(query_data) \
        .set_input_key(key_data) \
        .set_input_value(value_data) \
        .set_input_query_weight(query_weight_data) \
        .set_input_key_weight(key_weight_data) \
        .set_input_value_weight(value_weight_data) \
        .set_input_attn_mask(attn_mask_data) \
        .set_input_out_proj_weight(out_proj_weight_data) \
        .set_input_query_bias(query_bias_data) \
        .set_input_key_bias(key_bias_data) \
        .set_input_value_bias(value_bias_data) \
        .set_input_out_proj_bias(out_proj_bias_data) \
        .set_input_dropout_mask(dropout_mask_data) \
        .set_attr_attn_head_num(attn_head_num) \
        .set_attr_attn_dim_per_head(attn_dim_per_head) \
        .set_attr_src_len(src_len) \
        .set_attr_tgt_len(tgt_len) \
        .set_attr_keep_prob(keep_prob) \
        .set_attr_softmax_use_float(softmax_use_float);

    std::vector<Operator> inputs = {query_data, key_data, value_data, query_weight_data, key_weight_data,
        value_weight_data, attn_mask_data, out_proj_weight_data, query_bias_data, key_bias_data, 
        value_bias_data, out_proj_bias_data, dropout_mask_data
    };
    std::vector<Operator> outputs = {MultiHeadAttentionOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MultiHeadAttentionFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findMultiHeadAttention = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MultiHeadAttention") {
        findMultiHeadAttention = true;
        break;
    }
    }
    EXPECT_EQ(findMultiHeadAttention, true);
}
} // namespace