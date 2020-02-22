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

#include <exception>

#include "node_enum.hpp"

ngraph::op::OP_TYPEID ngraph::runtime::gpu::to_enum(const Node& node)
{
    return to_enum(&node);
}

ngraph::op::OP_TYPEID ngraph::runtime::gpu::to_enum(const Node* node)
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {"Abs", op::OP_TYPEID::Abs},
// {"Acos", op::OP_TYPEID::Acos},
// ...
    static std::unordered_map<std::string, op::OP_TYPEID> typeid_map{
#define NGRAPH_OP(a, b) {#a, ngraph::op::OP_TYPEID::a},
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
// #define NGRAPH_OP(a, b, c) {#a, op::OP_TYPEID::a},
// #include "op/op_tbl.hpp"
// #undef NGRAPH_OP
    };

    op::OP_TYPEID rc;
    auto it = typeid_map.find(node->description());
    if (it != typeid_map.end())
    {
        rc = it->second;
    }
    else
    {
        throw unsupported_op(node->description() + " not supported in to_enum");
    }
    return rc;
}
