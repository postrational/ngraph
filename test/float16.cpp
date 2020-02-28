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

#include <climits>
#include <random>

#include "gtest/gtest.h"

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/type/float16.hpp"
#include "util/float_util.hpp"

using namespace std;
using namespace ngraph;

TEST(float16, conversions)
{
    float16 f16;
    const char* source_string;
    std::string f16_string;

    // 1.f
    source_string = "0  01111  00 0000 0000";
    f16 = test::bits_to_float16(source_string);
    EXPECT_EQ(f16, float16(1.0));
    f16_string = test::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), 1.0);

    // -1.f
    source_string = "1  01111  00 0000 0000";
    f16 = test::bits_to_float16(source_string);
    EXPECT_EQ(f16, float16(-1.0));
    f16_string = test::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), -1.0);

    // 0.f
    source_string = "0  00000  00 0000 0000";
    f16 = test::bits_to_float16(source_string);
    EXPECT_EQ(f16, float16(0.0));
    f16_string = test::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), 0.0);

    // 1.5f
    source_string = "0  01111  10 0000 0000";
    f16 = test::bits_to_float16(source_string);
    EXPECT_EQ(f16, float16(1.5));
    f16_string = test::float16_to_bits(f16);
    EXPECT_STREQ(source_string, f16_string.c_str());
    EXPECT_EQ(static_cast<float>(f16), 1.5);
}

TEST(float16, assigns)
{
    float16 f16;
    f16 = 2.0;
    EXPECT_EQ(f16, float16(2.0));

    std::vector<float> f32vec{1.0, 2.0, 4.0};
    std::vector<float16> f16vec;
    std::copy(f32vec.begin(), f32vec.end(), std::back_inserter(f16vec));
    for (size_t i = 0; i < f32vec.size(); ++i)
    {
        EXPECT_EQ(f32vec.at(i), f16vec.at(i));
    }

    float f32arr[] = {1.0, 2.0, 4.0};
    float16 f16arr[sizeof(f32arr)];
    for (size_t i = 0; i < sizeof(f32arr) / sizeof(f32arr[0]); ++i)
    {
        f16arr[i] = f32arr[i];
        EXPECT_EQ(f32arr[i], f16arr[i]);
    }
}

struct DoubleF16
{
    double d;
    float16 f16;
};

TEST(float16, values)
{
    std::vector<DoubleF16> vals{{std::numeric_limits<double>::infinity(), {0, 0x1F, 0}},
                                {-std::numeric_limits<double>::infinity(), {1, 0x1F, 0}},
                                {std::numeric_limits<double>::quiet_NaN(), {0, 0x1F, 0x200}},
                                {std::numeric_limits<double>::signaling_NaN(), {0, 0x1f, 0x300}},
                                {2.73786e-05, float16::from_bits(459)},
                                {3.87722e-05, float16::from_bits(650)},
                                {-0.0223043, float16::from_bits(42422)},
                                {5.10779e-05, float16::from_bits(857)},
                                {-5.10779e-05, float16::from_bits(0x8359)},
                                {-2.553895e-05, float16::from_bits(0x81ac)},
                                {-0.0001021558, float16::from_bits(0x86b2)},
                                {5.960464477539063e-08, float16::from_bits(0x01)},
                                {8.940696716308594e-08, float16::from_bits(0x02)},
                                {65536.0, float16::from_bits(0x7c00)},
                                {65519.0, float16::from_bits(0x7bff)},
                                {65520.0, float16::from_bits(0x7c00)}};
    for (size_t i = 0; i < vals.size(); ++i)
    {
        auto df16 = vals.at(i);
        double d = df16.d;
        float16 f16 = d;
        EXPECT_EQ(df16.f16.to_bits(), f16.to_bits());
    }
}
