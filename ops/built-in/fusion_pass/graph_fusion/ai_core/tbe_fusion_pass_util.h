#ifndef FE_TBE_FUSION_PASS_UTIL_H
#define FE_TBE_FUSION_PASS_UTIL_H

#include <vector>
#include <string>

#include "securec.h"
#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
/**
 * Insert a transpose before one input of one op
 * @param [in] fusedNode which node will be inserted
 * @param [in] inputIndex which input index will be inserted
 * @param [in] permList transpose list
 * @param [in] graph
 * @return status whether insert success
 */
Status AddTransposeBeforeNode(const ge::NodePtr& fusedNode, const int64_t& inputIndex,
                              const vector<int64_t>& permList, ge::ComputeGraph &graph);

/**
 * Insert a transpose after one output of one op
 * @param [in] fusedNode which node will be inserted
 * @param [in] inputIndex which output index will be inserted
 * @param [in] permList transpose list
 * @param [in] graph
 * @return status whether insert success
 */

Status AddTransposeAfterNode(const ge::NodePtr& fusedNode, const int64_t& outputIndex,
                             const vector<int64_t>& permList, ge::ComputeGraph &graph);

class TbeFusionPassUtil {
 public:
/**
 * Get int type const value from tensor data
 * @param [in] data const tensor data
 * @param [in] data_type DT_INT8, DT_INT16, DT_INT32, DT_INT64
 * @param [out] const_values const int values
 * @return true:success, false:failed.
 */
  static bool GetConstIntData(const ge::Tensor& data, ge::DataType data_type,
                              std::vector<int64_t>& const_values);

  /**
 * Get int type const value from tensor data
 * @param [in] op Operator
 * @param [in] name name of the input
 * @param [out] values const int values
 * @return true:success, false:failed.
 */
  static bool GetConstIntData(const ge::Operator& op, const std::string& name,
                              std::vector<int64_t>& values);
};

}  // namespace fe

#endif  // FE_TBE_FUSION_PASS_UTIL_H
