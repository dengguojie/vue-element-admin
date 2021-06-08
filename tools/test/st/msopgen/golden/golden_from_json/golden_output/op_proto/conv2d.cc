#include "conv2d.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(Conv2DInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2D, Conv2DVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Conv2D, Conv2DInferShape);
VERIFY_FUNC_REG(Conv2D, Conv2DVerify);

}  // namespace ge
