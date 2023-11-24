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
        S = np.kron(np.eye(self.H), self.B)

        S = np.zeros((self.H * self.dx, self.H * self.du))
        M = np.zeros((self.H * self.dx, self.dx))

        for i in range(self.H-1):
            for j in range(i, self.H-1):
                S[(j+1) * self.dx : (j + 2) * self.dx, i * self.du : (i + 1) * self.du] = np.linalg.matrix_power(self.A, j +1 - i) @ self.B

        for i in range(self.H):
            M[i * self.dx : (i + 1) * self.dx, :] = np.linalg.matrix_power(self.A, i+1)

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
        G = -np.linalg.inv((S.T @ Qbar @ S + Rbar)) @ S.T @ Qbar @ M

        G0 = G[:self.du,:]
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

        Ginf = -np.linalg.inv(self.R + self.B.T @ theta_T_theta @ self.B) @ self.B.T @ theta_T_theta @ self.A

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

        U = cp.Variable((self.H, self.du))

        cost = cp.quad_form(S @ U + M @ x0.reshape((-1,1)), Qbar) + cp.quad_form(U, Rbar)

        constraints = [U <= u_max, u_min <= U]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve()

        U = U.value

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
        X = cp.Variable((self.H+1, self.dx))
        U = cp.Variable((self.H, self.du))

        cost = cp.sum(cp.sum_squares(X @ np.sqrt(self.Q.T)) + cp.sum_squares(U @ np.sqrt(self.R.T)))

        constraints = [X[0] == x0]
        constraints += [X[t+1] == self.A @ X[t] + self.B @ U[t] for t in range(self.H)]
        constraints += [u_min <= U, U <= u_max]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        U = U.value
        X = X.value

 
        # ---

        return U, X
