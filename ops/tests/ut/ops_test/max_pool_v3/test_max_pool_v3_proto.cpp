#include <gtest/gtest.h>
#include <vector>
#include "inc/nn_pooling_ops.h"

class MaxPoolV3Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "max_pool_v3 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "max_pool_v3 test TearDown" << std::endl;
    }
};

TEST_F(MaxPoolV3Test, max_pool_v3_test_case_1) {
    //  define your op here
     ge::op::MaxPoolV3 max_pool_v3_op;
     ge::TensorDesc tensorDesc;
     ge::Shape shape({1, 64, 56, 56});
     tensorDesc.SetDataType(ge::DT_FLOAT16);
     tensorDesc.SetShape(shape);

    //  update op input here
    max_pool_v3_op.UpdateInputDesc("x", tensorDesc);
    max_pool_v3_op.SetAttr("ksize",  {1, 1, 3, 3});
    max_pool_v3_op.SetAttr("strides",{1, 1, 2, 2});
    max_pool_v3_op.SetAttr("pads",{0, 0, 0, 0});
    max_pool_v3_op.SetAttr("global_pooling",false);
    max_pool_v3_op.SetAttr("ceil_mode",false);

    auto status = max_pool_v3_op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = max_pool_v3_op.InferShapeAndType();

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);


    // compare dtype and shape of op output
    auto output_desc = max_pool_v3_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 64, 28, 28};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
