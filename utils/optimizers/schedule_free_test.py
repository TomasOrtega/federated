from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from utils.optimizers import schedule_free


def schedule_free_update_numpy(var, grad, t, x, z, lr=0.01, beta=0.5):
    """Performs a Schedule-Free optimizer update using NumPy.

    Args:
        var: Current value of the variable (represents y_t).
        grad: Gradient computed at y_t.
        t: Current time step (starting from 1).
        x: Previous value of x_t.
        z: Previous value of z_t.
        lr: Learning rate.
        beta: Interpolation parameter beta.

    Returns:
        Updated values of var (y_t+1), x_t+1, z_t+1.
    """
    # Update z_{t+1} = z_t - lr * grad(y_t)
    z_t_plus_one = z - lr * grad

    # Update x_{t+1} = (1 - 1/(t+1)) * x_t + (1/(t+1)) * z_{t+1}
    one_over_t = 1.0 / (t + 1)
    x_t_plus_one = (1.0 - one_over_t) * x + one_over_t * z_t_plus_one

    # Compute y_{t+1} = (1 - beta) * z_{t+1} + beta * x_{t+1}
    y_t_plus_one = (1.0 - beta) * z_t_plus_one + beta * x_t_plus_one

    return y_t_plus_one, x_t_plus_one, z_t_plus_one


class ScheduleFreeOptimizerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters([tf.float32, tf.float64])
    def test_schedule_free_optimizer_dense(self, dtype):
        """Tests the Schedule-Free optimizer with dense gradients."""
        # Initialize variables and parameters
        initial_var = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grad = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        initial_x = np.zeros_like(initial_var)
        initial_z = np.zeros_like(initial_var)
        initial_y = (1.0 - 0.5) * initial_z + 0.5 * initial_x  # beta = 0.5
        learning_rate = 0.01
        beta = 0.5
        num_updates = 5

        # TensorFlow variables
        var_tf = tf.Variable(initial_value=initial_y,
                             dtype=dtype)  # var represents y_t
        optimizer = schedule_free.ScheduleFree(
            learning_rate=learning_rate, beta=beta)

        # Initialize slots x, z, y
        optimizer._create_slots([var_tf])
        x_tf = optimizer.get_slot(var_tf, 'x')
        z_tf = optimizer.get_slot(var_tf, 'z')
        y_tf = optimizer.get_slot(var_tf, 'y')

        # Assign initial values to x, z, y
        self.evaluate([
            x_tf.assign(initial_x),
            z_tf.assign(initial_z),
            y_tf.assign(initial_y),
            optimizer.t.assign(0)
        ])
        self.evaluate(tf.compat.v1.variables_initializer(
            optimizer.variables()))

        # Run updates and compare with NumPy implementation
        for t in range(num_updates):
            # Compute expected values using NumPy
            var_np, x_np, z_np = schedule_free_update_numpy(
                var=self.evaluate(var_tf),
                grad=grad,
                t=t,
                x=self.evaluate(x_tf),
                z=self.evaluate(z_tf),
                lr=learning_rate,
                beta=beta
            )

            # Apply gradients using the optimizer
            # Assume the gradient is computed at y_t (var_tf)
            optimizer.apply_gradients(
                [(tf.constant(grad, dtype=dtype), var_tf)])

            # Compare TensorFlow and NumPy values
            var_tf_val = self.evaluate(var_tf)
            x_tf_val = self.evaluate(x_tf)
            z_tf_val = self.evaluate(z_tf)

            np.testing.assert_allclose(var_tf_val, var_np, rtol=1e-6)
            np.testing.assert_allclose(x_tf_val, x_np, rtol=1e-6)
            np.testing.assert_allclose(z_tf_val, z_np, rtol=1e-6)

    @parameterized.parameters([tf.float32, tf.float64])
    def test_schedule_free_optimizer_sparse(self, dtype):
        """Tests the Schedule-Free optimizer with sparse gradients."""
        # Initialize variables and parameters
        initial_var = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)
        grad_values = np.array([[0.1], [0.1]], dtype=dtype.as_numpy_dtype)
        indices = np.array([0, 1], dtype=np.int32)
        initial_x = np.zeros_like(initial_var)
        initial_z = np.zeros_like(initial_var)
        initial_y = (1.0 - 0.5) * initial_z + 0.5 * initial_x  # beta = 0.5
        learning_rate = 0.01
        beta = 0.5
        num_updates = 5

        # TensorFlow variables
        var_tf = tf.Variable(initial_value=initial_y,
                             dtype=dtype)  # var represents y_t
        optimizer = schedule_free.ScheduleFree(
            learning_rate=learning_rate, beta=beta)

        # Initialize slots x, z, y
        optimizer._create_slots([var_tf])
        x_tf = optimizer.get_slot(var_tf, 'x')
        z_tf = optimizer.get_slot(var_tf, 'z')
        y_tf = optimizer.get_slot(var_tf, 'y')

        # Assign initial values to x, z, y
        self.evaluate([
            x_tf.assign(initial_x),
            z_tf.assign(initial_z),
            y_tf.assign(initial_y),
            optimizer.t.assign(0)
        ])
        self.evaluate(tf.compat.v1.variables_initializer(
            optimizer.variables()))

        # Create sparse gradient
        grad_tf = tf.IndexedSlices(
            values=tf.constant(grad_values, dtype=dtype),
            indices=tf.constant(indices),
            dense_shape=var_tf.shape
        )

        # Run updates and compare with NumPy implementation
        for t in range(num_updates):
            # Gather current variable, x, and z values
            var_current = self.evaluate(var_tf)
            x_current = self.evaluate(x_tf)
            z_current = self.evaluate(z_tf)

            # Compute expected values using NumPy
            var_np = var_current.copy()
            x_np = x_current.copy()
            z_np = z_current.copy()

            for idx, idx_val in enumerate(indices):
                var_np[idx_val], x_np[idx_val], z_np[idx_val] = schedule_free_update_numpy(
                    var=var_current[idx_val],
                    grad=grad_values[idx],
                    t=t,
                    x=x_current[idx_val],
                    z=z_current[idx_val],
                    lr=learning_rate,
                    beta=beta
                )

            # Apply sparse gradients using the optimizer
            optimizer.apply_gradients([(grad_tf, var_tf)])

            # Compare TensorFlow and NumPy values
            var_tf_val = self.evaluate(var_tf)
            x_tf_val = self.evaluate(x_tf)
            z_tf_val = self.evaluate(z_tf)

            np.testing.assert_allclose(var_tf_val, var_np, rtol=1e-6)
            np.testing.assert_allclose(x_tf_val, x_np, rtol=1e-6)
            np.testing.assert_allclose(z_tf_val, z_np, rtol=1e-6)

    def test_schedule_free_optimizer_config(self):
        """Tests that the optimizer configuration can be retrieved and set."""
        optimizer = schedule_free.ScheduleFree(learning_rate=0.01, beta=0.5)
        config = optimizer.get_config()
        new_optimizer = schedule_free.ScheduleFree.from_config(config)
        self.assertEqual(optimizer.get_config(), new_optimizer.get_config())


if __name__ == '__main__':
    tf.test.main()
