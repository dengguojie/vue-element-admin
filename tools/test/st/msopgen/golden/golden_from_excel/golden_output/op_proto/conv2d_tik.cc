#include "conv2d_tik.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(Conv2DTikInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTik, Conv2DTikVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Conv2DTik, Conv2DTikInferShape);
VERIFY_FUNC_REG(Conv2DTik, Conv2DTikVerify);

}  // namespace ge
