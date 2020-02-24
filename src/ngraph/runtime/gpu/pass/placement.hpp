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

#include <memory>
#include <set>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                class Placement;
            }
        }
    }
}

class ngraph::runtime::gpu::pass::Placement : public ngraph::pass::FunctionPass
{
public:
    Placement();

    enum class DeviceSupport
    {
        UNKNOWN,
        SUPPORTED,
        UNSUPPORTED
    };

    static DeviceSupport is_supported_on_device(std::shared_ptr<Node> node);
    static void assign_placement(std::shared_ptr<Node> node);

private:
    virtual bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};
