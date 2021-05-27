#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"



class GemmProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Gemm Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Gemm Proto Test TearDown" << std::endl;
  }
};


TEST_F(GemmProtoTest, GemmBaseTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_FLOAT16));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_FLOAT16));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));
   auto status = op.VerifyAllAttr(true);
   EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(GemmProtoTest, GemmAtypeTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_INT32));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_FLOAT16));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));
   auto status = op.VerifyAllAttr(true);
   EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GemmProtoTest, GemmBtypeTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_FLOAT16));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_INT32));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_FLOAT16));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));
   auto status = op.VerifyAllAttr(true);
   EXPECT_EQ(status, ge::GRAPH_FAILED);
}


TEST_F(GemmProtoTest, GemmCtypeTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_FLOAT16));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_INT8));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_FLOAT16));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));
   auto status = op.VerifyAllAttr(true);
   EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GemmProtoTest, GemmAlphatypeTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_FLOAT16));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_INT8));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));
   auto status = op.VerifyAllAttr(true);
   EXPECT_EQ(status, ge::GRAPH_FAILED);
}


TEST_F(GemmProtoTest, GemmBetatypeTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_FLOAT16));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_FLOAT16));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_INT8));
   auto status = op.VerifyAllAttr(true);
   EXPECT_EQ(status, ge::GRAPH_FAILED);
}



TEST_F(GemmProtoTest, GemmInferShapeTest) {
   ge::op::GEMM op;
   op.UpdateInputDesc("a", create_desc({32,64}, ge::DT_FLOAT16));
   op.UpdateInputDesc("b", create_desc({64,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("c", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateOutputDesc("y", create_desc({32,32}, ge::DT_FLOAT16));
   op.UpdateInputDesc("alpha", create_desc({1}, ge::DT_FLOAT16));
   op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));
   auto status = op.InferShapeAndType();
   EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

