#include "gtest/gtest.h"
#include <iostream>
#include "elewise_calculation_ops.h"
#include "math_ops.h"
#include "op_proto_test_util.h"

class SobolSample : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SobolSample SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SobolSample TearDown" << std::endl;
  }
};

TEST_F(SobolSample, SobolSample_test01) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_FLOAT);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SobolSample, SobolSample_test02) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SobolSample, SobolSample_infer_failed_1) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto dim_desc = op.GetInputDescByName("dim");
  EXPECT_EQ(dim_desc.GetDataType(), ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SobolSample, SobolSample_infer_failed_2) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_BOOL);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SobolSample, SobolSample_verify_failed_1) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("num_results", create_desc({3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto num_results_desc = op.GetInputDescByName("num_results");
  EXPECT_EQ(num_results_desc.GetShape().GetDimNum(), 2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SobolSample, SobolSample_verify_failed_2) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto num_results_desc = op.GetInputDescByName("num_results");
  EXPECT_EQ(num_results_desc.GetDataType(), ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SobolSample, SobolSample_verify_failed_3) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("skip", create_desc({3, 4}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto dim_desc = op.GetInputDescByName("dim");
  EXPECT_EQ(dim_desc.GetShape().GetDimNum(), 2);
  auto skip_desc = op.GetInputDescByName("skip");
  EXPECT_EQ(skip_desc.GetShape().GetDimNum(), 2);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(SobolSample, SobolSample_verify_failed_4) {
  ge::op::SobolSample op;
  op.UpdateInputDesc("dim", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("num_results", create_desc({}, ge::DT_FLOAT));
  op.UpdateInputDesc("skip", create_desc({}, ge::DT_FLOAT));
  op.SetAttr("dtype", ge::DT_DOUBLE);

  auto dim_desc = op.GetInputDescByName("dim");
  EXPECT_EQ(dim_desc.GetDataType(), ge::DT_FLOAT);
  auto num_results_desc = op.GetInputDescByName("num_results");
  EXPECT_EQ(num_results_desc.GetDataType(), ge::DT_FLOAT);
  auto skip_desc = op.GetInputDescByName("skip");
  EXPECT_EQ(skip_desc.GetDataType(), ge::DT_FLOAT);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
