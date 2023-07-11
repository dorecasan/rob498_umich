import control
import numpy as np
import scipy.linalg
import cvxpy as cp


class LinearMPC:

    def __init__(self, A, B, Q, R, horizon):
        self.dx = A.shape[0]
        self.du = B.shape[1]
        assert A.shape == (self.dx, self.dx)
        assert B.shape == (self.dx, self.du)
        self.H = horizon
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def compute_SM(self):
        """
        Computes the S and M matrices as defined in the ipython notebook

        All the variables you need should be class member variables already

        Returns:
            S: np.array of shape (horizon * dx, horizon * du) S matrix
            M: np.array of shape (horizon * dx, dx) M matrix

        """
        S, M = None, None

        # --- Your code here



        # ---
        return S, M

    def compute_Qbar_and_Rbar(self):
        Q_repeat = [self.Q] * self.H
        R_repeat = [self.R] * self.H
        return scipy.linalg.block_diag(*Q_repeat), scipy.linalg.block_diag(*R_repeat)

    def compute_finite_horizon_lqr_gain(self):
        """
            Compute the controller gain G0 for the finite-horizon LQR

        Returns:
            G0: np.array of shape (du, dx)

        """
        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()

        G0 = None

        # --- Your code here



        # ---

        return G0

    def compute_lqr_gain(self):
        """
            Compute controller gain G for infinite-horizon LQR
        Returns:
            Ginf: np.array of shape (du, dx)

        """
        Ginf = None
        theta_T_theta, _, _ = control.dare(self.A, self.B, self.Q, self.R)

        # --- Your code here



        # ---
        return Ginf

    def lqr_box_constraints_qp_shooting(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing with shooting

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls

        """

        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()
        U = None
        # --- Your code here



        # ---

        return U

    def lqr_box_constraints_qp_collocation(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing
            with collocation

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls
            X: np.array of shape (horizon, dx) containing sequence of optimal states

        """

        X, U = None, None

        # --- Your code here



        # ---

        return U, X
