#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UTIL_ANCHOR_UTIL_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UTIL_ANCHOR_UTIL_H_

#include <string.h>
#include "error_util.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "pattern_fusion_util.h"

inline ge::OutDataAnchorPtr GetPeerOutAnchorWithInDataAnchor(ge::NodePtr curr_node, int idx) {
  FUSION_PASS_CHECK(curr_node == nullptr, ge::CommonRuntimeErrLog("", "node is null."), return nullptr);

  auto in_anchor = curr_node->GetInDataAnchor(idx);
  FUSION_PASS_CHECK(in_anchor == nullptr, ge::CommonRuntimeErrLog("", "in data anchor is null."), return nullptr);

  return in_anchor->GetPeerOutAnchor();
}

inline ge::NodePtr GetPeerOutNodeWithInDataAnchor(ge::NodePtr curr_node, int idx) {
  auto peer_out_anchor = GetPeerOutAnchorWithInDataAnchor(curr_node, idx);
  FUSION_PASS_CHECK(peer_out_anchor == nullptr, ge::CommonRuntimeErrLog("", "peer out anchor is null."),
                    return nullptr);

  return peer_out_anchor->GetOwnerNode();
}

inline ge::InDataAnchorPtr GetPeerInAnchorByOutDataAnchor(ge::OutDataAnchorPtr out_anchor_ptr, size_t idx) {
  FUSION_PASS_CHECK(out_anchor_ptr == nullptr, ge::CommonRuntimeErrLog("", "out anchor is null."), return nullptr);

  auto peer_in_anchors = out_anchor_ptr->GetPeerInDataAnchors();
  FUSION_PASS_CHECK(peer_in_anchors.size() <= idx, ge::CommonRuntimeErrLog("", "idx out of range."), return nullptr);

  return peer_in_anchors.at(idx);
}

inline ge::NodePtr GetPeerInNodeByOutDataAnchor(ge::OutDataAnchorPtr out_anchor_ptr, size_t idx) {
  auto peer_in_anchor = GetPeerInAnchorByOutDataAnchor(out_anchor_ptr, idx);
  FUSION_PASS_CHECK(peer_in_anchor == nullptr, ge::CommonRuntimeErrLog("", "peer in anchor is null."), return nullptr);

  // assert: anchor in peer_in_anchors != nullptr
  return peer_in_anchor->GetOwnerNode();
}

inline ge::ConstGeTensorDescPtr GetCurrNodeInputDesc(ge::NodePtr curr_node, size_t idx) {
  auto curr_desc = curr_node->GetOpDesc();
  FUSION_PASS_CHECK(curr_desc == nullptr, ge::CommonRuntimeErrLog("", "current opdesc is null."), return nullptr);
  return curr_desc->GetInputDescPtr(idx);
}

inline ge::ConstGeTensorDescPtr GetCurrNodeOutputDesc(ge::NodePtr curr_node, size_t idx) {
  auto curr_desc = curr_node->GetOpDesc();
  FUSION_PASS_CHECK(curr_desc == nullptr, ge::CommonRuntimeErrLog("", "current opdesc is null."), return nullptr);
  return curr_desc->GetOutputDescPtr(idx);
}

template<typename T>
inline ge::GeTensorDescPtr GetCurrNodeMutableInputDesc(ge::NodePtr curr_node, T idx) {
  auto curr_desc = curr_node->GetOpDesc();
  FUSION_PASS_CHECK(curr_desc == nullptr, ge::CommonRuntimeErrLog("", "current opdesc is null."), return nullptr);
  return curr_desc->MutableOutputDesc(idx);
}

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UTIL_ANCHOR_UTIL_H_
