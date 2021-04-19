#ifndef FUSION_ENGINE_STUB_FUSIONPASSUTILS_H
#define FUSION_ENGINE_STUB_FUSIONPASSUTILS_H

#include "graph/compute_graph.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"


using std::string;
namespace fe
{

    class OpKernelStoreDefaultMock : public ge::OpsKernelInfoStore {
    public:
        OpKernelStoreDefaultMock()
        {
        };

        OpKernelStoreDefaultMock(const OpKernelStoreDefaultMock &) = delete;

        OpKernelStoreDefaultMock &operator=(const OpKernelStoreDefaultMock &) = delete;

        // initialize opsKernelInfoStore
        Status Initialize(const map<string, string> &options) override;

        // close opsKernelInfoStore
        ge::Status Finalize() override;

        // get all opsKernelInfo
        void GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const override;

        // whether the opsKernelInfoStore is supported based on the operator attribute
        bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &un_supported_reason) const override;

    };

    class FusionPassTestUtils {
    public:
        static Status
        RunGraphFusionPass(string fusion_pass_name, GraphFusionPassType passType, ge::ComputeGraph &computeGraph);
        static Status
        RunGraphFusionPass(string fusion_pass_name, GraphFusionPassType passType, ge::ComputeGraph &computeGraph, bool checkSupportedResult);
        static Status
        InferShapeAndType(ge::ComputeGraphPtr computeGraphPtr);
    };

}


#endif //FUSION_ENGINE_STUB_FUSIONPASSUTILS_H
