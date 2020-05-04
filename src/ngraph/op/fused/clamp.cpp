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
#include "ngraph/op/fused/clamp.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/runtime/reference/clamp.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Clamp::type_info;

namespace
{
    template <element::Type_t ET, typename T>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, T min, T max, size_t count)
    {
        runtime::reference::clamp<T>(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), min, max, count);
        return true;
    }

    bool evaluate_clamp(
        const HostTensorPtr& arg, const HostTensorPtr& out, double min, double max, size_t count)
    {
        auto ceil_func = [](double x) { return std::ceil(x); };
        auto floor_func = [](double x) { return std::floor(x); };

        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(i8)
            (arg,
             out,
             double_to_int<int8_t>(min, ceil_func),
             double_to_int<int8_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(i16)
            (arg,
             out,
             double_to_int<int16_t>(min, ceil_func),
             double_to_int<int16_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(i32)
            (arg,
             out,
             double_to_int<int32_t>(min, ceil_func),
             double_to_int<int32_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(i64)
            (arg,
             out,
             double_to_int<int64_t>(min, ceil_func),
             double_to_int<int64_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(u8)
            (arg,
             out,
             double_to_int<uint8_t>(min, ceil_func),
             double_to_int<uint8_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(u16)
            (arg,
             out,
             double_to_int<uint16_t>(min, ceil_func),
             double_to_int<uint16_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(u32)
            (arg,
             out,
             double_to_int<uint32_t>(min, ceil_func),
             double_to_int<uint32_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(u64)
            (arg,
             out,
             double_to_int<uint64_t>(min, ceil_func),
             double_to_int<uint64_t>(max, floor_func),
             count);
            break;
            TYPE_CASE(f32)(arg, out, static_cast<float>(min), static_cast<float>(max), count);
            break;
            TYPE_CASE(f64)(arg, out, min, max, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Clamp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_clamp(
        inputs[0], outputs[0], get_min(), get_max(), shape_size(get_output_shape(0)));
}

op::Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : FusedOp({data})
    , m_min{min}
    , m_max{max}
{
    constructor_validate_and_infer_types();
}

void op::Clamp::pre_validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(
        this, m_min < m_max, "The 'min' parameter needs to be less than 'max' for Clamp");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

NodeVector op::Clamp::decompose_op() const
{
    const auto data = input_value(0);
    auto data_type = data.get_element_type();
    const auto data_shape = data.get_shape();

    auto ceil_func = [](double x) { return std::ceil(x); };
    auto floor_func = [](double x) { return std::floor(x); };

    switch (get_element_type())
    {
        case element::Type_t::i8:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<int8_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<int8_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::i16:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<int16_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<int16_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::i32:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<int32_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<int32_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::i64:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<int64_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<int64_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::u8:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<uint8_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<uint8_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::u16:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<uint16_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<uint16_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::u32:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<uint32_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<uint32_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::u64:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, double_to_int<uint64_t>(m_min, ceil_func));
            auto clamp_max = builder::make_constant(data_type, data_shape, double_to_int<uint64_t>(m_max, floor_func));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::f32:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, static_cast<float>(m_min));
            auto clamp_max = builder::make_constant(data_type, data_shape, static_cast<float>(m_max));
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        case element::Type_t::f64:
        {
            auto clamp_min = builder::make_constant(data_type, data_shape, m_min);
            auto clamp_max = builder::make_constant(data_type, data_shape, m_max);
            return {std::make_shared<ngraph::op::Minimum>(clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
            break;
        }
        default:
            // TODO: error
            break;
    }
}

shared_ptr<Node> op::Clamp::clone_with_new_inputs(const OutputVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the Clamp op but got ",
                          new_args.size());

    return make_shared<Clamp>(new_args.at(0), m_min, m_max);
}

bool op::Clamp::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    return true;
}
