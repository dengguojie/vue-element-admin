#ifndef THE_AIPP_CONV_FUSION_RELU_H
#define THE_AIPP_CONV_FUSION_RELU_H
#include <string>
#include "graph/node.h"

namespace fe {

class TbeAippFusionRule {
public:
	static bool CheckAippConvStridehValidation(ge::NodePtr convNode);
	static bool CheckConvload2dNodeValidation(ge::NodePtr convNode);
	static bool CheckAippConvEltwiseFusionValidation(ge::NodePtr convNode, const string &inputFormat);

};
} // namespace
#endif  // THE_AIPP_CONV_FUSION_RELU_H