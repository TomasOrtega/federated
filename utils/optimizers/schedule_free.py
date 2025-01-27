import tensorflow as tf

class ScheduleFree(tf.keras.optimizers.legacy.Optimizer):
    """Optimizer that implements the Schedule-Free learning algorithm.
    
    See https://github.com/facebookresearch/schedule_free
    
    The Schedule-Free optimizer replaces the momentum of an underlying optimizer
    with a combination of interpolation and averaging, without requiring a
    decreasing learning rate schedule.

    The update equations are:

        y_t = (1 - beta) * z_t + beta * x_t
        z_{t+1} = z_t - lr * grad(y_t)
        x_{t+1} = (1 - 1/(t+1)) * x_t + (1/(t+1)) * z_{t+1}

    Attributes:
        learning_rate: A float value or a constant float tensor. The learning rate.
        beta: A float value or a constant float tensor. The interpolation parameter.
        name: Optional name for the operations created when applying gradients.
        **kwargs: Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
          `decay`}.
    """

    def __init__(self, learning_rate=0.01, beta=0.5, name='ScheduleFree', **kwargs):
        """Construct a new Schedule-Free optimizer.

        Args:
            learning_rate: A float value or a constant float tensor. The learning rate.
            beta: A float value or a constant float tensor. The interpolation parameter.
            name: Optional name for the operations created when applying gradients.
                Defaults to "ScheduleFree".
            **kwargs: Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
              `decay`}.
        """
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta', beta)
        # Initialize time step t
        self.t = None  # Will be created in _create_slots

    def _create_slots(self, var_list):
        """Create slots for x_t, z_t, and y_t for each variable."""
        for var in var_list:
            self.add_slot(var, 'x')  # Slot to store x_t
            self.add_slot(var, 'z')  # Slot to store z_t
            self.add_slot(var, 'y')  # Slot to store y_t

        # Create a non-trainable variable to track the time step t
        self.t = self.add_weight(
            name='iter', shape=(), initializer='zeros', trainable=False, dtype=tf.int32)

    def _resource_apply_dense(self, grad, var):
        """Apply gradients to variables.

        Args:
            grad: A `Tensor` representing the gradient.
            var: A `Variable` object to be updated.
        """
        var_dtype = var.dtype.base_dtype
        lr = self._decayed_lr(var_dtype)
        beta = self._get_hyper('beta', var_dtype)

        # Update time step t
        t = self.t.assign_add(1)
        t_float = tf.cast(t, var_dtype)

        # Get the slots for x_t, z_t, y_t
        x = self.get_slot(var, 'x')
        z = self.get_slot(var, 'z')
        y = self.get_slot(var, 'y')  # y_t

        # Update z_{t+1} = z_t - lr * grad(y_t)
        z_t_plus_one = z.assign_sub(lr * grad, use_locking=self._use_locking)

        # Update x_{t+1} = (1 - 1/(t+1)) * x_t + (1/(t+1)) * z_{t+1}
        one_over_t = 1.0 / t_float
        x_t_plus_one = x.assign(
            (1.0 - one_over_t) * x + one_over_t * z_t_plus_one,
            use_locking=self._use_locking)

        # Compute y_{t+1} = (1 - beta) * z_{t+1} + beta * x_{t+1}
        y_t_plus_one = y.assign(
            (1.0 - beta) * z_t_plus_one + beta * x_t_plus_one,
            use_locking=self._use_locking)

        # Update the variable to y_{t+1} for the next iteration
        var_update = var.assign(y_t_plus_one, use_locking=self._use_locking)

        return tf.group(*[var_update, x_t_plus_one, z_t_plus_one, y_t_plus_one])

    def _resource_apply_sparse(self, grad, var, indices):
        """Apply gradients to variables when the gradient is sparse.

        Args:
            grad: A `Tensor` representing the gradient values for the indices.
            var: A `Variable` object to be updated.
            indices: A `Tensor` of indices into the first dimension of `var` and `grad`.
        """
        var_dtype = var.dtype.base_dtype
        lr = self._decayed_lr(var_dtype)
        beta = self._get_hyper('beta', var_dtype)

        # Update time step t
        t = self.t.assign_add(1)
        t_float = tf.cast(t, var_dtype)

        # Get the slots for x_t, z_t, y_t
        x = self.get_slot(var, 'x')
        z = self.get_slot(var, 'z')
        y = self.get_slot(var, 'y')  # y_t

        # Update z_{t+1} at the specified indices
        z_slice = tf.gather(z, indices)
        grad_slice = grad

        z_t_plus_one_slice = z_slice - lr * grad_slice
        z_t_plus_one = self._resource_scatter_update(z, indices, z_t_plus_one_slice)

        # Update x_{t+1}
        x_slice = tf.gather(x, indices)
        one_over_t = 1.0 / t_float
        x_t_plus_one_slice = (1.0 - one_over_t) * x_slice + one_over_t * z_t_plus_one_slice
        x_t_plus_one = self._resource_scatter_update(x, indices, x_t_plus_one_slice)

        # Compute y_{t+1}
        y_t_plus_one_slice = (1.0 - beta) * z_t_plus_one_slice + beta * x_t_plus_one_slice
        y_t_plus_one = self._resource_scatter_update(y, indices, y_t_plus_one_slice)

        # Update the variable to y_{t+1} at the specified indices
        var_update = self._resource_scatter_update(var, indices, y_t_plus_one_slice)

        return tf.group(*[var_update, x_t_plus_one, z_t_plus_one, y_t_plus_one])

    def get_config(self):
        """Returns the config of the optimizer.

        Returns:
            A dictionary containing the configuration of the optimizer.
        """
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta': self._serialize_hyperparameter('beta'),
        })
        return config