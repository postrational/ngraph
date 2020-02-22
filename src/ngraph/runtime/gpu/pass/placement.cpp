//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <numeric>

#include "ngraph/runtime/gpu/pass/placement.hpp"

using namespace ngraph;
using namespace std;

runtime::nnp::pass::OpPlacement::OpPlacement()
{
}

bool runtime::nnp::pass::OpPlacement::run_on_function(std::shared_ptr<ngraph::Function> function)
{
    for (shared_ptr<Node> node : function->get_ops())
    {
        assign_placement(node);
    }
    return false;
}

void runtime::nnp::pass::OpPlacement::assign_placement(shared_ptr<Node> node)
{
    if (is_supported_on_device(node) != DeviceSupport::SUPPORTED)
    {
        node->set_placement(Placement::CPU);
    }
    else
    {
        node->set_placement(Placement::DEFAULT);
    }
}

runtime::nnp::pass::OpPlacement::DeviceSupport
    runtime::nnp::pass::OpPlacement::is_supported_on_device(shared_ptr<Node> node)
{
    DeviceSupport rc = DeviceSupport::UNKNOWN;
    try
    {
        bool support = true;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
        switch (to_enum(node.get()))
        {
        // case OP_TYPEID::Acos: support = false; break;
        // case OP_TYPEID::All: support = false; break;
        // case OP_TYPEID::Any: support = false; break;
        // case OP_TYPEID::ArgMax: support = false; break;
        // case OP_TYPEID::ArgMin: support = false; break;
        // case OP_TYPEID::Asin: support = false; break;
        // case OP_TYPEID::Atan: support = false; break;
        // case OP_TYPEID::Atan2: support = false; break;
        // case OP_TYPEID::AvgPool: support = is_avg_pool_supported(node); break;
        // case OP_TYPEID::AvgPoolBackprop: support = is_avg_pool_backprop_supported(node); break;
        // case OP_TYPEID::BatchNormInference:
        // case OP_TYPEID::BatchNormTraining: support = is_batch_norm_supported(node); break;
        // case OP_TYPEID::BatchNormTrainingBackprop:
        //     support = is_batch_norm_backprop_supported(node);
        //     break;
        // case OP_TYPEID::Broadcast: support = is_broadcast_supported(node); break;
        // case OP_TYPEID::Ceiling: support = false; break;
        // case OP_TYPEID::Concat: support = is_concat_supported(node); break;
        // case OP_TYPEID::Convolution: support = is_convolution_supported(node); break;
        // case OP_TYPEID::ConvolutionBackpropData:
        //     support = is_convolution_backprop_data_supported(node);
        //     break;
        // case OP_TYPEID::ConvolutionBackpropFilters:
        //     support = is_convolution_backprop_filters_supported(node);
        //     break;
        // case OP_TYPEID::Cos: support = false; break;
        // case OP_TYPEID::CumSum: support = false; break;
        // case OP_TYPEID::Dot: support = is_dot_supported(node); break;
        // case OP_TYPEID::DynBroadcast: support = false; break;
        // case OP_TYPEID::DynPad: support = false; break;
        // case OP_TYPEID::DynReplaceSlice: support = false; break;
        // case OP_TYPEID::DynReshape: support = false; break;
        // case OP_TYPEID::DynSlice: support = false; break;
        // case OP_TYPEID::EmbeddingLookup: support = false; break;
        // case OP_TYPEID::Floor: support = false; break;
        // case OP_TYPEID::Gather: support = is_gather_supported(node); break;
        // case OP_TYPEID::GatherND: support = false; break;
        // case OP_TYPEID::LRN: support = false; break;
        // case OP_TYPEID::LayerNormBackprop: support = is_layernorm_bprop_supported(node); break;
        // case OP_TYPEID::Maximum: support = is_maximum_supported(node); break;
        // case OP_TYPEID::MaxPool: support = is_max_pool_supported(node); break;
        // case OP_TYPEID::MaxPoolBackprop: support = is_max_pool_backprop_supported(node); break;
        // case OP_TYPEID::Pad: support = is_pad_supported(node); break;
        // case OP_TYPEID::Product: support = false; break;
        // case OP_TYPEID::RandomUniform: support = false; break;
        // case OP_TYPEID::ReplaceSlice: support = is_replace_slice_supported(node); break;
        // case OP_TYPEID::Reverse: support = is_reverse_supported(node); break;
        // case OP_TYPEID::ReverseSequence: support = false; break;
        // case OP_TYPEID::Round: support = false; break;
        // case OP_TYPEID::ScatterAdd: support = is_scatter_add_supported(node); break;
        // case OP_TYPEID::ScatterNDAdd: support = false; break;
        // case OP_TYPEID::Select: support = false; break;
        // case OP_TYPEID::ShapeOf: support = false; break;
        // case OP_TYPEID::Sign: support = false; break;
        // case OP_TYPEID::Sin: support = false; break;
        // case OP_TYPEID::Sinh: support = false; break;
        // case OP_TYPEID::Slice: support = is_slice_supported(node); break;
        // case OP_TYPEID::Softmax: support = is_softmax_supported(node); break;
        // case OP_TYPEID::Tan: support = false; break;
        // case OP_TYPEID::TopK: support = is_topk_supported(node); break;
        }
#pragma GCC diagnostic pop
        rc = (support ? DeviceSupport::SUPPORTED : DeviceSupport::UNSUPPORTED);
    }
    catch (...)
    {
    }
    if (rc == DeviceSupport::UNSUPPORTED)
    {
        NGRAPH_DEBUG << "Placing on host: " << *node;
    }
    return rc;
}
