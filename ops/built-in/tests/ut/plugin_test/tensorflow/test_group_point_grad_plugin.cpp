#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"
#include "plugin_test_utils.h"

class group_point_grad_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "group_point_grad_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "group_point_grad_test TearDown" << std::endl;
  }
};

TEST_F(group_point_grad_test, group_point_grad_test_1) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/gatherpointgrad_case_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}