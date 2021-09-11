#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"


class CosineSimilarityTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cosine_similarity test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "cosine_similarity test TearDown" << std::endl;
    }
};

TEST_F(CosineSimilarityTest, cosine_similarity_test_case_1) {
     ge::op::CosineSimilarity cosine_similarity_op;
     ge::TensorDesc tensor_desc;
     ge::Shape shape({2, 3, 4});
     tensor_desc.SetDataType(ge::DT_FLOAT);
     tensor_desc.SetShape(shape);
     tensor_desc.SetOriginShape(shape);
     cosine_similarity_op.UpdateInputDesc("input_x1", tensor_desc);
     cosine_similarity_op.UpdateInputDesc("input_x2", tensor_desc);

     auto ret = cosine_similarity_op.InferShapeAndType();
     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

     auto output_desc = cosine_similarity_op.GetOutputDesc("output_y");
     EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
     std::vector<int64_t> expected_output_shape = {2, 4};
     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
