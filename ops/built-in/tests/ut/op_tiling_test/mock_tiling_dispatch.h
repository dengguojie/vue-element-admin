#ifndef OPTILING_UT_MOCK_TILING_DISPATCH_H_
#define OPTILING_UT_MOCK_TILING_DISPATCH_H_

namespace optiling {
bool MockEletwiseTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info) {
  return true;
}
bool MockBroadcastTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info) {
  return true;
}
bool MockReduceTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info) {
  return true;
}
bool MockNormTiling(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                    utils::OpRunInfo& run_info) {
  return true;
}
bool MockTransposeDsl(const std::string& op_type, const ge::Operator& op_paras, const nlohmann::json& op_info,
                      utils::OpRunInfo& run_info) {
  return true;
}
}

#define ReduceTiling MockReduceTiling
#define NormTiling MockNormTiling
#define EletwiseTiling MockEletwiseTiling
#define BroadcastTiling MockBroadcastTiling
#define TransposeDsl MockTransposeDsl

#endif // OPTILING_UT_MOCK_TILING_DISPATCH_H_
