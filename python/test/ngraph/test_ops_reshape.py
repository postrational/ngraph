# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import pytest

import ngraph as ng
from test.ngraph.util import get_runtime, run_op_node, run_op_numeric_data


@pytest.mark.skip_on_gpu
def test_concat():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    axis = 0
    expected = np.concatenate((a, b), axis=0)

    runtime = get_runtime()
    parameter_a = ng.parameter(list(a.shape), name='A', dtype=np.float32)
    parameter_b = ng.parameter(list(b.shape), name='B', dtype=np.float32)
    node = ng.concat([parameter_a, parameter_b], axis)
    computation = runtime.computation(node, parameter_a, parameter_b)
    result = computation(a, b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type, value', [
    (bool, False),
    (bool, np.empty((2, 2), dtype=bool)),
])
@pytest.mark.skip_on_gpu  # under investigation, runtime error is: function failed to compile
def test_constant_from_bool(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type, value', [
    (np.float32, np.float32(0.1234)),
    (np.float64, np.float64(0.1234)),
    (np.int8, np.int8(-63)),
    (np.int16, np.int16(-12345)),
    (np.int32, np.int32(-123456)),
    (np.int64, np.int64(-1234567)),
    (np.uint8, np.uint8(63)),
    (np.uint16, np.uint16(12345)),
    (np.uint32, np.uint32(123456)),
    (np.uint64, np.uint64(1234567)),
])
@pytest.mark.skip_on_gpu  # under investigation, runtime error is: function failed to compile
def test_constant_from_scalar(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type', [
    np.float32,
    np.float64,
])
@pytest.mark.skip_on_gpu  # under investigation, runtime error is: function failed to compile
def test_constant_from_float_array(val_type):
    np.random.seed(133391)
    input_data = np.array(-1 + np.random.rand(2, 3, 4) * 2, dtype=val_type)
    result = run_op_numeric_data(input_data, ng.constant, val_type)
    assert np.allclose(result, input_data)


@pytest.mark.parametrize('val_type, range_start, range_end', [
    (np.int8, -8, 8),
    (np.int16, -64, 64),
    (np.int32, -1024, 1024),
    (np.int64, -16383, 16383),
    (np.uint8, 0, 8),
    (np.uint16, 0, 64),
    (np.uint32, 0, 1024),
    (np.uint64, 0, 16383),
])
@pytest.mark.skip_on_gpu  # under investigation, runtime error is: function failed to compile
def test_constant_from_integer_array(val_type, range_start, range_end):
    np.random.seed(133391)
    input_data = np.array(np.random.randint(range_start, range_end, size=(2, 2)), dtype=val_type)
    result = run_op_numeric_data(input_data, ng.constant, val_type)
    assert np.allclose(result, input_data)


@pytest.mark.skip_on_gpu
def test_broadcast_numpy():
    input_tensor = np.array([1.0, 2.0, 3.0], np.float32)
    input_shape = np.array([3, 3], np.int64)
    input_axes = np.array([1], np.int64)
    result = run_op_node([input_tensor, input_shape, input_axes], ng.broadcast)

    a_arr = np.array([[0], [0], [0]], dtype=np.float32)
    b_arr = np.array([[1, 2, 3]], dtype=np.float32)
    expected = np.add(a_arr, b_arr)

    assert np.allclose(result, expected)

@pytest.mark.skip_on_gpu
def test_broadcast_bidirectional():
    input_tensor = np.array([1.0, 2.0, 3.0], np.float32)
    input_shape = np.array([3, 3], np.int64)
    input_axes = np.array([1], np.int64)
    result = run_op_node([input_tensor, input_shape, input_axes], ng.broadcast, 'BIDIRECTIONAL')

    print(result)
    a_arr = np.array([[0], [0], [0]], dtype=np.float32)
    b_arr = np.array([[1, 2, 3]], dtype=np.float32)
    expected = np.add(a_arr, b_arr)

    assert np.allclose(result, expected)


@pytest.mark.skip_on_gpu
def test_transpose():
    input_tensor = np.arange(3 * 3 * 224 * 224).reshape((3, 3, 224, 224))
    input_order = np.array([0, 2, 3, 1])

    result = run_op_node([input_tensor, input_order], ng.transpose)

    expected = np.transpose(input_tensor, input_order)

    assert np.allclose(result, expected)


@pytest.mark.skip_on_gpu
@pytest.mark.skip_on_interpreter  # unsupported op
def test_tile():
    input_tensor = np.arange(6).reshape((2, 1, 3))
    repeats = np.array([2, 1])

    result = run_op_node([input_tensor, repeats], ng.tile)

    expected = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]).reshape((2, 2, 3))

    assert np.allclose(result, expected)


@pytest.mark.skip_on_gpu
def test_strided_slice():
    input_tensor = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    begin = np.array([1, 0])
    end = np.array([0, 0])
    strides = np.array([1, 1])
    begin_mask = np.array([0, 0, 0])
    end_mask = np.array([0, 0, 0])
    new_axis_mask = np.array([0, 1, 0])
    shrink_axis_mask = np.array([1, 0, 0])
    ellipsis_mask = np.array([0, 0, 0])

    result = run_op_node([input_tensor, begin, end, strides],
                         ng.strided_slice,
                         begin_mask,
                         end_mask,
                         new_axis_mask,
                         shrink_axis_mask,
                         ellipsis_mask)

    expected = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]).reshape((1, 3, 4))

    assert np.allclose(result, expected)


@pytest.mark.skip_on_gpu
def test_reshape_v1():
    A = np.arange(1200, dtype=np.float32).reshape((2, 5, 5, 24))
    shape = np.array([0, -1, 4])
    special_zero = True

    expected_shape = np.array([2, 150, 4])
    expected = np.reshape(A, expected_shape)
    result = run_op_node([A, shape], ng.reshape, special_zero)

    assert np.allclose(result, expected)


@pytest.mark.skip_on_gpu
def test_shape_of():
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    result = run_op_node([input_tensor], ng.shape_of)

    assert np.allclose(result, [3, 3])
