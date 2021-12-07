#include "fusion_pass_test_utils.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"

namespace fe {

    ge::Status OpKernelStoreDefaultMock::Initialize(const map<string, string> &options) {
        return SUCCESS;
    }

    ge::Status OpKernelStoreDefaultMock::Finalize() {
        return SUCCESS;
    }

    bool
    OpKernelStoreDefaultMock::CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &un_supported_reason) const {
        return true;
    }

    void OpKernelStoreDefaultMock::GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const {

    }

    OpsKernelInfoStorePtr defaultOpKernelStore = make_shared<OpKernelStoreDefaultMock>();

    class falseCheckOpKernelStoreMock : public OpKernelStoreDefaultMock {
    public:

        bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string &un_supported_reason) const {
            return false;
        }

    };

    OpsKernelInfoStorePtr falseCheckSupportedOpKernelStore = make_shared<falseCheckOpKernelStoreMock>();


    Status FusionPassTestUtils::RunGraphFusionPass(string fusionPassName, GraphFusionPassType passType,
                                                   ge::ComputeGraph &computeGraph) {
        std::map<string, FusionPassRegistry::CreateFn> createFns =
                FusionPassRegistry::GetInstance().GetCreateFnByType(passType);
        const auto &iter = createFns.find(fusionPassName);
        if (iter != createFns.end()) {
            if (passType == fe::BUILT_IN_GRAPH_PASS || passType == fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS ||
                passType == fe::BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS) {
                auto graphFusionPassBasePtr = std::unique_ptr<PatternFusionBasePass>(
                        dynamic_cast<PatternFusionBasePass *>(iter->second()));
                if (graphFusionPassBasePtr == nullptr) {
                    return FAILED;
                }
                graphFusionPassBasePtr->SetName(fusionPassName);
                auto ret = graphFusionPassBasePtr->Run(computeGraph, defaultOpKernelStore);
                return ret;
            } else {
                auto graphFusionPassBasePtr = std::unique_ptr<GraphFusionPassBase>(
                        dynamic_cast<GraphFusionPassBase *>(iter->second()));
                if (graphFusionPassBasePtr == nullptr) {
                    return FAILED;
                }
                graphFusionPassBasePtr->SetName(fusionPassName);
                auto ret = graphFusionPassBasePtr->Run(computeGraph);
                return ret;
            }

        }
        return FAILED;

    }

    Status FusionPassTestUtils::RunGraphFusionPass(string fusionPassName, GraphFusionPassType passType,
                                                   ge::ComputeGraph &computeGraph, bool checkSupportedResult) {
        std::map<string, FusionPassRegistry::CreateFn> createFns =
                FusionPassRegistry::GetInstance().GetCreateFnByType(passType);
        const auto &iter = createFns.find(fusionPassName);
        if (iter != createFns.end()) {
            if (passType == fe::BUILT_IN_GRAPH_PASS || passType == fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS ||
                passType == fe::BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS) {
                auto graphFusionPassBasePtr = std::unique_ptr<PatternFusionBasePass>(
                        dynamic_cast<PatternFusionBasePass *>(iter->second()));
                if (graphFusionPassBasePtr == nullptr) {
                    return FAILED;
                }
                graphFusionPassBasePtr->SetName(fusionPassName);
                if (checkSupportedResult) {
                    auto ret = graphFusionPassBasePtr->Run(computeGraph, defaultOpKernelStore);
                    return ret;
                } else {
                    auto ret = graphFusionPassBasePtr->Run(computeGraph, falseCheckSupportedOpKernelStore);
                    return ret;
                }
            } else {
                auto graphFusionPassBasePtr = std::unique_ptr<GraphFusionPassBase>(
                        dynamic_cast<GraphFusionPassBase *>(iter->second()));
                if (graphFusionPassBasePtr == nullptr) {
                    return FAILED;
                }
                graphFusionPassBasePtr->SetName(fusionPassName);
                auto ret = graphFusionPassBasePtr->Run(computeGraph);
                return ret;
            }

        }
        return FAILED;

    }

    Status FusionPassTestUtils::InferShapeAndType(ge::ComputeGraphPtr computeGraphPtr) {
        computeGraphPtr->TopologicalSorting();
        for (auto nodePtr: computeGraphPtr->GetAllNodes()) {
            if (nodePtr->GetType() != "Data") {
                auto verifyStatus = nodePtr->Verify();
                if (verifyStatus != ge::SUCCESS) {
                    std::cout << "Graph Infer failed, " << nodePtr->GetName()
                              << "'s Verify() failed" << std::endl;
                    return verifyStatus;
                }
                auto inferFormatStatus = nodePtr->InferOriginFormat();
                if (inferFormatStatus != ge::SUCCESS) {
                    std::cout << "Graph Infer failed, " << nodePtr->GetName()
                              << "'s InferOriginFormat() failed" << std::endl;
                    return inferFormatStatus;
                }
                auto inferShapeStatus = nodePtr->InferShapeAndType();
                if (inferShapeStatus != ge::SUCCESS) {
                    std::cout << "Graph Infer failed, " << nodePtr->GetName()
                              << "'s inferShapeAndType() failed" << std::endl;
                    return inferShapeStatus;
                }
            }
            for (auto outDataAnchor : nodePtr->GetAllOutAnchors()) {
                int outIdx = outDataAnchor->GetIdx();
                auto output_desc = nodePtr->GetOpDesc()->MutableOutputDesc(outIdx);
                if (output_desc->GetOriginShape().GetShapeSize() == 0 and output_desc->GetShape().GetShapeSize() != 0) {
                    output_desc->SetOriginFormat(output_desc->GetFormat());
                    output_desc->SetOriginShape(output_desc->GetShape());
                }
                for (auto anchor : outDataAnchor->GetPeerAnchors()) {
                    int idx = anchor->GetIdx();
                    auto update_status = anchor->GetOwnerNode()->GetOpDesc()
                            ->UpdateInputDesc(idx, nodePtr->GetOpDesc()->GetOutputDesc(outIdx)
                            );
                    if (update_status != ge::SUCCESS) {
                        std::cout << "Graph Infer failed, update " << anchor->GetOwnerNode()->GetName()
                                  << "'s input failed" << std::endl;
                        return update_status;
                    }
                }
            }
        }
        return SUCCESS;
    }
}
