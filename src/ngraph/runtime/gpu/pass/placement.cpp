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

#include "ngraph/log.hpp"
#include "ngraph/runtime/gpu/node_enum.hpp"
#include "ngraph/runtime/gpu/pass/placement.hpp"

using namespace ngraph;
using namespace std;

runtime::gpu::pass::Placement::Placement()
{
}

bool runtime::gpu::pass::Placement::run_on_function(std::shared_ptr<ngraph::Function> function)
{
    for (shared_ptr<Node> node : function->get_ops())
    {
        assign_placement(node);
    }
    return false;
}

void runtime::gpu::pass::Placement::assign_placement(shared_ptr<Node> node)
{
    if (is_supported_on_device(node) != DeviceSupport::SUPPORTED)
    {
        NGRAPH_DEBUG << "Placing on host: " << *node;
        node->set_placement(ngraph::Placement::CPU);
    }
    else
    {
        NGRAPH_DEBUG << "Placing on device: " << *node;
        node->set_placement(ngraph::Placement::DEFAULT);
    }
}

runtime::gpu::pass::Placement::DeviceSupport
    runtime::gpu::pass::Placement::is_supported_on_device(shared_ptr<Node> node)
{
    using TYPEID = ngraph::op::OP_TYPEID;
    DeviceSupport rc = DeviceSupport::UNKNOWN;
    try
    {
        bool support = true;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
        switch (to_enum(node.get()))
        {
        case TYPEID::EmbeddingLookup: support = false; break;
        }
#pragma GCC diagnostic pop
        rc = (support ? DeviceSupport::SUPPORTED : DeviceSupport::UNSUPPORTED);
    }
    catch (...)
    {
    }
    return rc;
}
