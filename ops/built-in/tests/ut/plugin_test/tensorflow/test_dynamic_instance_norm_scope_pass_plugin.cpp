#include <string>
#include <vector>
#include <map>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class dyncmic_instance_norm_scope_pass_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dyncmic_instance_norm_scope_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dyncmic_instance_norm_scope_test TearDown" << std::endl;
  }
};

TEST_F(dyncmic_instance_norm_scope_pass_plugin_test, dyncmic_instance_norm_scope_pass_plugin_test_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/instance_norm_1.pb";
  std::map<ge::AscendString, ge::AscendString> params;
  string key = "enable_scope_fusion_passes";
  string value = "ScopeInstanceNormPass";
  params.insert(std::make_pair(ge::AscendString(key.c_str()), ge::AscendString(value.c_str())));

  // auto status = aclgrphParseTensorFlow(modelFile.c_str(), params, graph);
}
