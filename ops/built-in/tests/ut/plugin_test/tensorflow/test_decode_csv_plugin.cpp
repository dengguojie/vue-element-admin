#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"

using namespace ge;

class decode_csv_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "decode_csv_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "decode_csv_plugin_test TearDown" << std::endl;
  }
};

TEST_F(decode_csv_plugin_test, decode_csv_plugin_test_1) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/decode_csv_case_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}