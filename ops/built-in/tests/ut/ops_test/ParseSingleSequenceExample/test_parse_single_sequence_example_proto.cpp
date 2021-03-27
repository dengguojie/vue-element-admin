#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "parsing_ops.h"

using namespace ge;
using namespace op;
using std::vector;
using std::string;

class parse_single_sequence_example : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "parse_single_sequence_example SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "parse_single_sequence_example TearDown" << std::endl;
    }
};

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_01) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//Ncontext_sparse
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_02) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//Ncontext_dense
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_03) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_04) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_05) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_6) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({1}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_7) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_08) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_DOUBLE};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_09) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_DOUBLE};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//feature_list_dense_shapes
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_10) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//Tcontext_dense
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_11) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//feature_list_sparse_types
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_12) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//feature_list_dense_types
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_13) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<vector<int64_t>> context_dense_shapes {{-1}};
    op.SetAttr("context_dense_shapes", context_dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//context_dense_shapes
TEST_F(parse_single_sequence_example, parse_single_sequence_example_infer_shape_14) {
    ge::op::ParseSingleSequenceExample op;

    op.SetAttr("Ncontext_sparse", 1);
    op.SetAttr("Ncontext_dense", 1);

    vector<vector<int64_t>> feature_list_dense_shapes {{-1}};
    op.SetAttr("feature_list_dense_shapes", feature_list_dense_shapes);

    op.SetAttr("Nfeature_list_sparse", 1);
    op.SetAttr("Nfeature_list_dense", 1);

    vector<DataType> Tcontext_dense {DT_FLOAT};
    op.SetAttr("Tcontext_dense", Tcontext_dense);

    vector<DataType> feature_list_sparse_types {DT_FLOAT};
    op.SetAttr("feature_list_sparse_types", feature_list_sparse_types);

    vector<DataType> feature_list_dense_types {DT_FLOAT};
    op.SetAttr("feature_list_dense_types", feature_list_dense_types);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    op.UpdateInputDesc("feature_list_dense_missing_assumed_empty", create_desc({-1}, DT_STRING));
    op.UpdateInputDesc("debug_name", create_desc({-1}, DT_STRING));

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}