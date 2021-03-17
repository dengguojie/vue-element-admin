#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class EyeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "eye test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "eye test TearDown" << std::endl;
    }
};

TEST_F(EyeTest, eye_test_case_1) {
     ge::op::Eye eye_op;
    
    int num_rows_value = 3;
    eye_op.SetAttr("num_rows", num_rows_value);
    int num_columns_value = 3;
    eye_op.SetAttr("num_columns", num_columns_value);
    std::vector<int> batch_shape_value = {1,};
    eye_op.SetAttr("batch_shape", batch_shape_value);
    int dtype_value = 0;
    eye_op.SetAttr("dtype", dtype_value);

     auto ret = eye_op.InferShapeAndType();
     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

     auto output_desc = eye_op.GetOutputDesc("y");
     EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
     std::vector<int64_t> expected_output_shape = {1, 3, 3};
     EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
