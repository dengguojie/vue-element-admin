#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "linalg_ops.h"

class GerTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Ger SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Ger TearDown" << std::endl;
    }
};

TEST_F(GerTest, ger_test_case_1) {
    ge::op::Ger gerOp;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({10, });
    tensorDesc1.SetDataType(ge::DT_FLOAT16);
    tensorDesc1.SetShape(shape1);
    gerOp.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({20, });
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    gerOp.UpdateInputDesc("vec2", tensorDesc2);
	
    auto ret = gerOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = gerOp.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {10, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_2) {
    ge::op::Ger gerOp;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({10, });
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    gerOp.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({20, });
    tensorDesc2.SetDataType(ge::DT_FLOAT);
    tensorDesc2.SetShape(shape2);
    gerOp.UpdateInputDesc("vec2", tensorDesc2);
	
    auto ret = gerOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = gerOp.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {10, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_3) {
    ge::op::Ger gerOp;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({1, });
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    gerOp.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({20, });
    tensorDesc2.SetDataType(ge::DT_FLOAT);
    tensorDesc2.SetShape(shape2);
    gerOp.UpdateInputDesc("vec2", tensorDesc2);
	
    auto ret = gerOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = gerOp.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {1, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(GerTest, ger_test_case_4) {
    ge::op::Ger gerOp;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({10, });
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    gerOp.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({20, });
    tensorDesc2.SetDataType(ge::DT_FLOAT16);
    tensorDesc2.SetShape(shape2);
    gerOp.UpdateInputDesc("vec2", tensorDesc2);

    auto ret = gerOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GerTest, ger_test_case_5) {
    ge::op::Ger gerOp;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({10, 10});
    tensorDesc1.SetDataType(ge::DT_FLOAT);
    tensorDesc1.SetShape(shape1);
    gerOp.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({10, 20});
    tensorDesc2.SetDataType(ge::DT_FLOAT);
    tensorDesc2.SetShape(shape2);
    gerOp.UpdateInputDesc("vec2", tensorDesc2);

    auto ret = gerOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(GerTest, ger_test_case_6) {
    ge::op::Ger gerOp;

    ge::TensorDesc tensorDesc1;
    ge::Shape shape1({10, });
    tensorDesc1.SetDataType(ge::DT_INT32);
    tensorDesc1.SetShape(shape1);
    gerOp.UpdateInputDesc("x", tensorDesc1);

    ge::TensorDesc tensorDesc2;
    ge::Shape shape2({20, });
    tensorDesc2.SetDataType(ge::DT_INT32);
    tensorDesc2.SetShape(shape2);
    gerOp.UpdateInputDesc("vec2", tensorDesc2);

    auto ret = gerOp.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}