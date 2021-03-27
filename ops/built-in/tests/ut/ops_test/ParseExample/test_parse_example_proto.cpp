#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "parsing_ops.h"

using namespace ge;
using namespace op;

using std::vector;

class parse_example : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "parse_example SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "parse_example TearDown" << std::endl;
    }
};

TEST_F(parse_example, parse_example_infer_shape_01) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(parse_example, parse_example_infer_shape_02) {
    ge::op::ParseExample op;

    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_example, parse_example_infer_shape_03) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//sparse_types
TEST_F(parse_example, parse_example_infer_shape_04) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//Tdense
TEST_F(parse_example, parse_example_infer_shape_05) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

//dense_shapes
TEST_F(parse_example, parse_example_infer_shape_06) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_example, parse_example_infer_shape_07) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({1}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_example, parse_example_infer_shape_08) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", -1);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_example, parse_example_infer_shape_09) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", -1);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_example, parse_example_infer_shape_10) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_DOUBLE, DT_DOUBLE};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_FLOAT, DT_FLOAT};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(parse_example, parse_example_infer_shape_11) {
    ge::op::ParseExample op;

    op.SetAttr("Nsparse", 2);
    op.SetAttr("Ndence", 2);
    vector<DataType> sparse_types {DT_FLOAT, DT_FLOAT};
    op.SetAttr("sparse_types", sparse_types);
    vector<DataType> Tdense {DT_DOUBLE, DT_DOUBLE};
    op.SetAttr("Tdense", Tdense);
    vector<vector<int64_t>> dense_shapes {{-1}, {-1}};
    op.SetAttr("dense_shapes", dense_shapes);

    op.UpdateInputDesc("serialized", create_desc({}, DT_STRING));
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}