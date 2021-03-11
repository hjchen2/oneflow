/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

namespace {

bool IsReadyToRun(const std::vector<std::shared_ptr<TensorArg>>& out_grads) {
  return std::any_of(
      out_grads.begin(), out_grads.end(),
      [](const std::shared_ptr<TensorArg>& tensor_arg) { return !tensor_arg->Empty(); });
}

Maybe<void> InitEmptyTensorArgs2ZerosTensor(
    const TensorTuple& outputs, const std::vector<std::shared_ptr<TensorArg>>& out_grads) {
  for (int i = 0; i < out_grads.size(); i++) {
    if (out_grads[i]->Empty()) {
      TODO();  // wangyinggang: out_grads[i]->PushPartialTensor(Tensor.zeros_like(outputs[i]));
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

StackFunctionNode::StackFunctionNode(
    const std::shared_ptr<const std::function<Maybe<void>()>>& backward_fn,
    const TensorTuple& inputs, const TensorTuple& outputs) {
  inputs_ = std::make_shared<TensorTuple>(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    inputs_->at(i) = inputs[i];
    in_grads_.emplace_back(inputs[i]->now_grad_arg());
  }

  outputs_ = std::make_shared<TensorTuple>(outputs.size());
  for (int i = 0; i < outputs.size(); i++) {
    TODO();  // shares data with output tensors but not grad_fn
    out_grads_.emplace_back(outputs[i]->now_grad_arg());
  }

  backward_fn_ = backward_fn;
}

void StackFunctionNode::ReleaseOutTensorArgs() {
  for (const std::shared_ptr<TensorArg>& tensor_arg : out_grads_) { tensor_arg->Release(); }
}

void StackFunctionNode::ReleaseGraph() {
  inputs_.reset();
  outputs_.reset();
  in_grads_.clear();
  out_grads_.clear();
  backward_fn_.reset();
}

Maybe<void> StackFunctionNode::Apply(bool create_graph) {
  CHECK_OR_RETURN(!backward_fn_) << "This FunctionNode with name `" << GetOpName() << "` has been released.";
  if (!IsReadyToRun(out_grads_)) { return Maybe<void>::Ok(); }
  InitEmptyTensorArgs2ZerosTensor(*outputs_, out_grads_);
  TODO();  // wangyinggang: Calls backward_fn_ and passes arguments according to AutogradInterpreter
  return Maybe<void>::Ok();
}

Maybe<TensorTuple> StackAutogradEngine::Execute(const TensorTuple& outputs,
                                                const TensorTuple& inputs,
                                                const TensorTuple& out_grads, bool retain_graph,
                                                bool create_graph) {
  bool is_capture_grads = !inputs.empty();
  std::shared_ptr<TensorTuple> captured_tensors = std::make_shared<TensorTuple>(inputs.size());
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i]->now_grad_arg()->PushPartialTensor(out_grads[i]);
  }
  auto it = node_list_.begin();
  while (it != node_list_.end()) {
    // Skips node when already released
    if (!it->lock()) {
      node_list_.erase(it);
      continue;
    }
    JUST(it->lock()->Apply(create_graph));
    if (is_capture_grads) {
      TODO();  // wangyinggang: Captures grads in out_grads
    } else {
      TODO();  // wangyinggang: Accumulates grads for leaf or retain_grad tensors
    }
    it->lock()->ReleaseOutTensorArgs();
    if (retain_graph) {
      it++;
    } else {
      it->lock()->ReleaseGraph();
      node_list_.erase(it);
    }
  }
  return captured_tensors;
}

const std::shared_ptr<FunctionNode>& StackAutogradEngine::AddBackwardFuncPtr(
    const std::shared_ptr<const std::function<void()>>& backward_fn, const TensorTuple& inputs,
    TensorTuple& outputs) {
  std::shared_ptr<FunctionNode> func_node =
      std::make_shared<StackFunctionNode>(backward_fn, inputs, outputs);
  for (const std::shared_ptr<Tensor>& out_tensor : outputs) {
    out_tensor->set_grad_fn_node(func_node);
  }
  node_list_.push_front(func_node);
  return std::move(func_node);
}

}  // namespace one
}  // namespace oneflow
