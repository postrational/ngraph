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

#pragma once

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        enum class OP_TYPEID;
    }
    namespace runtime
    {
        namespace gpu
        {
            ngraph::op::OP_TYPEID to_enum(const Node* node);
            ngraph::op::OP_TYPEID to_enum(const Node& node);
        }
    }
}

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
enum class ngraph::op::OP_TYPEID
{
#define NGRAPH_OP(a, b) a,
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
    // #define NGRAPH_OP(a, b, c) a,
    // #include "op/op_tbl.hpp"
    // #undef NGRAPH_OP
};
