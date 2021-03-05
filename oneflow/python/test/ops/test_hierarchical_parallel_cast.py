"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
import numpy as np
import oneflow as flow


# test 1D
def _test(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_train(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x,
            parallel_hierarchy=[4],
            parallel_distribution=["S(1)"],
            grad_mode="manual",
            grad_parallel_hierarchy=[2, 2],
            grad_parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x,
            parallel_hierarchy=[4],
            parallel_distribution=["B"],
            grad_mode="manual",
            grad_parallel_hierarchy=[4],
            grad_parallel_distribution=["S(1)"],
        )
        x = flow.math.relu(x)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_gather(test_case):
    flow.clear_default_session()
    flow.config.collective_boxing.nccl_enable_all_to_all(True)
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(
        x: flow.typing.Numpy.Placeholder((1024, 4)),
        indices: flow.typing.Numpy.Placeholder(shape=(4,), dtype=flow.int32),
    ) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
        )
        indices = flow.hierarchical_parallel_cast(
            indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"]
        )
        x = flow.gather(x, indices)
        x = flow.hierarchical_parallel_cast(
            x,
            parallel_hierarchy=[2, 2],
            parallel_distribution=["S(0)", "S(0)"],
            name="cast_0",
        )
        # x = flow.hierarchical_parallel_cast(
        #    x,
        #    parallel_hierarchy=[2, 2],
        #    parallel_distribution=["S(1)", "S(0)"],
        #    name="cast_0",
        # )
        # x = flow.hierarchical_parallel_cast(
        #    x,
        #    parallel_hierarchy=[2, 2],
        #    parallel_distribution=["S(0)", "S(0)"],
        #    name="cast_1",
        # )
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 4).astype(np.float32)
    indices = np.random.randint(low=0, high=1024, size=(4,))
    y_arr = test_fn(x_arr, indices)
    gather_out = x_arr[indices]
    print("y_arr", y_arr.shape, y_arr)
    print("gather_out", gather_out.shape, gather_out)
    diff = y_arr.flatten() - gather_out.flatten()
    print("diff", diff[np.where(diff != 0)])
    test_case.assertTrue(np.allclose(y_arr.flatten(), gather_out.flatten()))


def _test_gather_train(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(
        indices: flow.typing.Numpy.Placeholder(shape=(12,), dtype=flow.int32),
    ) -> flow.typing.Numpy:
        x = flow.get_variable(
            name="x",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        indices = flow.hierarchical_parallel_cast(
            indices, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"]
        )
        x = flow.gather(x, indices)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(1)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    indices = np.random.randint(low=0, high=20, size=(12,))
    y_arr = test_fn(x_arr, indices)
    gather_out = x_arr[indices]
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("gather_out", gather_out.shape, gather_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), gather_out.flatten()))


def _test_slice(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        print(x.shape)
        x = flow.slice(x, begin=(None, 1), size=(None, x.shape[1] - 1))
        # x = flow.slice(x, begin=(None, None), size=(None, None))
        print(x.shape)
        # x = flow.hierarchical_parallel_cast(
        #    x, parallel_hierarchy=[2, 2], parallel_distribution=["B","S(1)"]
        # )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    slice_out = x_arr[:, :]
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("slice_out", slice_out.shape, slice_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), slice_out.flatten()))


def _test_layer_norm(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        print(x.shape)
        x = flow.slice(x, begin=(None, 1), size=(None, x.shape[1] - 1))
        # x = flow.slice(x, begin=(None, None), size=(None, None))
        print(x.shape)
        # x = flow.hierarchical_parallel_cast(
        #    x, parallel_hierarchy=[2, 2], parallel_distribution=["B","S(1)"]
        # )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    slice_out = x_arr[:, :]
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("slice_out", slice_out.shape, slice_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), slice_out.flatten()))


# test 2D axis 1 change
def _test0(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((512, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,s0]->(2,2)[s0,s1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(1)"],
        )
        x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        # (2, 2)[s0,p]->(2,2)[s0,B]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,B]->(2,2)[s0,S0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(512, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("x_arr", x_arr.sum(1).flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.sum(1).flatten()))


# test 2D axis 0 change
def _test1(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
        )
        x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(0)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x,
            parallel_hierarchy=[4],
            parallel_distribution=["S(0)"],
            grad_mode="manual",
            grad_parallel_hierarchy=[2, 2],
            grad_parallel_distribution=["S(0)", "S(0)"],
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("y_arr sum", y_arr.sum())
    print("x_arr", x_arr.sum(1).flatten()[0:10])
    print("x_arr sum", x_arr.sum())

    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.sum(1).flatten()))


# axis 01
def _test01(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        # 121
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2,2)[s0,s0]->(2,2)[s1,s0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
        )
        # x = flow.hierarchical_parallel_cast(
        #    x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        # )
        x = flow.math.relu(x)
        # (2,2)[s1,s0]->(2,2)[B,S1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2,2)[B,S1]->[S0,S0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"]
        )
        x = flow.math.relu(x)
        # (2,2)[S0,S0]->[4]S0
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("y_arr shape", y_arr.shape)
    print("x_arr", x_arr.flatten()[0:10])
    print("x_arr shape", x_arr.shape)
    test_case.assertTrue(np.allclose(y_arr, x_arr))


# axis 01
def _test_hie(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 512)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(1)"],
        )
        # x = flow.math.reduce_sum(x, axis=[1], keepdims=True)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "S(1)"],
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 512).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("y_arr sum", y_arr.sum())
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    print("x_arr sum", x_arr.sum())

    test_case.assertTrue(np.allclose(y_arr, x_arr))


# test 2D axis 1 change
def _testtest(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((512, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,s0]->(2,2)[s0,s1]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(0)"],
        )
        # (2, 2)[s0,p]->(2,2)[s0,B]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(1)", "S(1)"],
        )
        x = flow.math.relu(x)
        # (2, 2)[s0,B]->(2,2)[s0,S0]
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(512, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.flatten()[0:10])
    print("x_arr", x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_train1(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(1)"]
        )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        x = flow.math.relu(x)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(x)
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_train2(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("train", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["B", "B"],
        )
        print("xxx", x.shape)

        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["S(1)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        print("v", v.shape)
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        print("xxx", x.shape)

        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )

        x = flow.math.relu(x)
        y = flow.reshape(x, shape=(4, 256, -1))
        # print("x2", x.shape)

        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
        ).minimize(y)
        return y

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("x_arr", x_arr.shape, x_arr.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), x_arr.flatten()))


def _test_reshape(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "B"]
        )
        print(x.shape)
        x = flow.reshape(x, (512, 2048))
        print(x.shape)
        # x = flow.hierarchical_parallel_cast(
        #    x, parallel_hierarchy=[2, 2], parallel_distribution=["B","S(1)"]
        # )
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["B"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr)
    y_out = x_arr.reshape(512, 2048)
    print("y_arr", y_arr.shape, y_arr.flatten()[0:10])
    print("reshape_out", y_out.shape, y_out.flatten()[0:10])
    test_case.assertTrue(np.allclose(y_arr.flatten(), y_out.flatten()))


@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        # _test(test_case)
        # _test0(test_case)
        # _test1(test_case)
        # _test01(test_case)
        # _test_hie(test_case)
        # _testtest(test_case)
        # _test_gather(test_case)
        # _test_train2(test_case)
        # _test_slice(test_case)
        # _test_gather_train(test_case)
        _test_reshape(test_case)


if __name__ == "__main__":
    unittest.main()
