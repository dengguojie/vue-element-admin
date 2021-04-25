#include <gtest/gtest.h>

#include <iostream>

#include "lookup_ops.h"
#include "op_proto_test_util.h"

class LookupTableExportTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LookupTableExportTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LookupTableExportTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(LookupTableExportTest, LookupTableExportTest_Tkeys_error) {
  ge::op::LookupTableExport op;
  auto inference_context =
      std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  op.SetInferenceContext(inference_context);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableExportTest, LookupTableExportTest_handle_error) {
  ge::op::LookupTableExport op;
  op.UpdateInputDesc("handle", create_desc({1}, ge::DT_FLOAT16));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableExportTest, LookupTableExportTest_Tvalues_error) {
  ge::op::LookupTableExport op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.SetAttr("Tkeys", ge::DT_FLOAT);
  auto inference_context =
      std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  op.SetInferenceContext(inference_context);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LookupTableExportTest, LookupTableExportTest_success) {
  ge::op::LookupTableExport op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.SetAttr("Tkeys", ge::DT_FLOAT);
  op.SetAttr("Tvalues", ge::DT_FLOAT);
  auto inference_context =
      std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  op.SetInferenceContext(inference_context);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
