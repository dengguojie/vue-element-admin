#include "graph/operator_reg.h"

namespace ge {

REG_OP(ReFormat)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(ReFormat)
}
