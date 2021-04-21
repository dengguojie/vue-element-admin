#ifndef OPS_BUILT_IN_OP_PROTO_TOP_K_H_
#define OPS_BUILT_IN_OP_PROTO_TOP_K_H_
#include "graph/operator_reg.h"

namespace ge {
REG_OP(TopK)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(largest, Bool, true)
    .ATTR(dim, Int, -1)
    .OP_END_FACTORY_REG(TopK)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_TOP_K_H_
