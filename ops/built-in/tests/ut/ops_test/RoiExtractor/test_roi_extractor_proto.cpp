#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class RoiExtractorTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "roi_extractor test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "roi_extractor test TearDown" << std::endl;
  }
};

TEST_F(RoiExtractorTest, roi_extractor_test_case_1) {
  ge::op::RoiExtractor roi_extractor_op;
  roi_extractor_op.create_dynamic_input_features(4);
  roi_extractor_op.UpdateDynamicInputDesc("features", 0, create_desc({1, 16, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateDynamicInputDesc("features", 1, create_desc({1, 16, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateDynamicInputDesc("features", 2, create_desc({1, 16, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateDynamicInputDesc("features", 3, create_desc({1, 16, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateInputDesc("rois", create_desc({100, 5}, ge::DT_FLOAT));
  roi_extractor_op.set_attr_pooled_height(7);
  roi_extractor_op.set_attr_pooled_width(7);

  auto ret = roi_extractor_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = roi_extractor_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {100, 16, 7, 7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RoiExtractorTest, roi_extractor_test_case_2) {
  ge::op::RoiExtractor roi_extractor_op;
  roi_extractor_op.create_dynamic_input_features(4);
  roi_extractor_op.UpdateDynamicInputDesc("features", 0, create_desc({1, 256, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateDynamicInputDesc("features", 1, create_desc({1, 256, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateDynamicInputDesc("features", 2, create_desc({1, 256, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateDynamicInputDesc("features", 3, create_desc({1, 256, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateInputDesc("rois", create_desc({100, 5}, ge::DT_FLOAT));

  auto status = roi_extractor_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = roi_extractor_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = roi_extractor_op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {100, 256, 7, 7};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(RoiExtractorTest, roi_extractor_test_case_3) {
  ge::op::RoiExtractor roi_extractor_op;
  roi_extractor_op.create_dynamic_input_features(1);
  roi_extractor_op.UpdateDynamicInputDesc("features", 0, create_desc({1, 256, 3, 4}, ge::DT_FLOAT));
  roi_extractor_op.UpdateInputDesc("rois", create_desc({100, 5}, ge::DT_FLOAT));
  roi_extractor_op.SetAttr("pooled_height", true);
  roi_extractor_op.SetAttr("pooled_width", true);

  auto ret = roi_extractor_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}