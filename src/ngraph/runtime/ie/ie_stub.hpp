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

#include <iostream>
#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"

namespace InferenceEngine
{
    class CNNNetwork
    {
    };

    class ExecutableNetwork
    {
    };

    class Core
    {
    };

    class InferRequest
    {
    };

    class InputsDataMap
    {
    };

    class MemoryBlob
    {
    public:
        using Ptr = std::shared_ptr<MemoryBlob>;
    };

    class Blob
    {
    public:
        using Ptr = std::shared_ptr<Blob>;
    };

    using SizeVector = std::vector<size_t>;

    enum class Layout
    {
        SCALAR,
        C,
        NC,
        CHW,
        NCHW,
        NCDHW,
        GOIDHW
    };

    enum class Precision
    {
        BOOL,
        I8,
        I16,
        I32,
        I64,
        U8,
        U16,
        U32,
        U64,
        FP32,
    };

    class TensorDesc
    {
    public:
        TensorDesc(Precision, InferenceEngine::SizeVector, InferenceEngine::Layout);
    };

    template<typename T>
    class TBlob : public MemoryBlob
    {
    public:
        TBlob(TensorDesc);
    };
}

#define THROW_IE_EXCEPTION std::cout
