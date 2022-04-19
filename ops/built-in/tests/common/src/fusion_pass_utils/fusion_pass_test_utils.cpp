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
                passType == fe::BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS||
                passType == fe::BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS) {
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
                passType == fe::BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS ||
                passType == fe::BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS) {
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

    Status FusionPassTestUtils::RunBufferFusionPass(BufferFusionPassBase *ptr_buffer_fusion_pass_func,
                                                    const vector<BufferFusionPattern *> patterns,
                                                    ge::ComputeGraphPtr &compute_graph_ptr,
                                                    const BufferFusionMapping &mapping, const string &name_pattern) {
        for (const auto &pattern : patterns) {
            if (name_pattern != "" and pattern->GetName() != name_pattern) {
                continue;
            }

            if (mapping.empty()) {
                std::cout << "BufferFusionMapping is empty" << std::endl;
                continue;
            }

            if (mapping.size() != pattern->GetOpDescs().size()) {
                std::cout << "mapping size(" << mapping.size() << ") not match desc size("
                          << pattern->GetOpDescs().size() << ") in pattern(" << pattern->GetName() << ")" << std::endl;
                continue;
            } else {
                std::cout << "match mapping size with pattern(" << pattern->GetName() << ")" << std::endl;
            }
            bool match_pattern = true;
            for (const auto desc : pattern->GetOpDescs()) {
                auto ptr_nodes_match_desc = const_cast<BufferFusionMapping &>(mapping)[desc];
                if (ptr_nodes_match_desc.size() < desc->repeate_min or
                    ptr_nodes_match_desc.size() > desc->repeate_max) {
                    std::cout << desc->desc_name << " repeat num not match, real is " << ptr_nodes_match_desc.size()
                              << std::endl;
                    match_pattern = false;
                    break;
                }
            }

            if (match_pattern) {
                vector<ge::NodePtr> fusion_nodes;
                auto result = ptr_buffer_fusion_pass_func->GetFusionNodes(mapping, fusion_nodes);
                if (result == fe::SUCCESS) {
                    return fe::SUCCESS;
                }
            }
        }
        return fe::FAILED;
    }

    bool FusionPassTestUtils::GetBufferFusionPattern(const vector<BufferFusionPattern *> patterns,
                                                     const string &name_pattern, BufferFusionPattern **pattern) {
        for (auto curr_pattern : patterns) {
            if (curr_pattern->GetName() == name_pattern) {
                *pattern = curr_pattern;
                return true;
            }
        }

        return false;
    }
}
