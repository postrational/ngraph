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

#include "function_call.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::FunctionCall::type_info;

op::v0::FunctionCall::FunctionCall(const vector<Output<Node>>& outputs,
                                   const vector<Output<Node>>& inputs,
                                   const Function& function,
                                   shared_ptr<runtime::Backend> backend)
    : Op(inputs)
    , m_function_outputs{outputs}
    , m_function{ngraph::clone_function(function)}
    , m_backend{backend}
    , m_executable{backend->compile(m_function)}
{
    set_output_size(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++)
    {
        set_output_type(i, outputs[i].get_element_type(), outputs[i].get_partial_shape());
    }
}

const string& op::v0::FunctionCall::description() const
{
    static string s_type = "FunctionCall";
    return s_type;
}

shared_ptr<Node> op::v0::FunctionCall::copy_with_new_args(const NodeVector& new_args) const
{
    vector<Output<Node>> inputs;
    for (const shared_ptr<Node>& arg : new_args)
    {
        inputs.push_back(arg);
    }
    return make_shared<FunctionCall>(m_function_outputs, inputs, *m_function, m_backend);
}

void op::v0::FunctionCall::set_function_outputs(const std::vector<Output<Node>>& function_outputs)
{
    m_function_outputs = function_outputs;
}

std::vector<Output<Node>> op::v0::FunctionCall::get_function_outputs() const
{
    return m_function_outputs;
}

void op::v0::FunctionCall::set_backend(const shared_ptr<runtime::Backend>& backend)
{
    m_backend = backend;
}

shared_ptr<runtime::Backend> op::v0::FunctionCall::get_backend() const
{
    return m_backend;
}

void op::v0::FunctionCall::set_executable(const shared_ptr<runtime::Executable>& executable)
{
    m_executable = executable;
}

shared_ptr<runtime::Executable> op::v0::FunctionCall::get_executable() const
{
    return m_executable;
}

void op::v0::FunctionCall::set_function(const shared_ptr<Function>& function)
{
    m_function = function;
}

shared_ptr<Function> op::v0::FunctionCall::get_function() const
{
    return m_function;
}
