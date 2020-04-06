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

#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"

#ifdef NGRAPH_INTERPRETER_ENABLE
namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            bool init()
            {
                ngraph_register_interpreter_backend();
            }
            static bool s_interpreter_init = init();
        }
    }
}
#endif
