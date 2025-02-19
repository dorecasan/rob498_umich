import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym


class CartpoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cartpole = None
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (4,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (4,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        self.cartpole = p.loadURDF('cartpole.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (4,) representing cartpole state [x, theta, x_dot, theta_dot]

        """

        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot = p.getJointState(self.cartpole, 1)[0:2]
        return np.array([x, theta, x_dot, theta_dot])

    def set_state(self, state):
        x, theta, x_dot, theta_dot = state
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta, targetVelocity=theta_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(low=np.array([x_lims[0], theta_lims[0], x_dot_lims[0], theta_dot_lims[0]]),
                                     high=np.array([x_lims[1], theta_lims[1], x_dot_lims[1], theta_dot_lims[
                                         1]]))  # linear force # TODO: Verifty that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 240
        self.render_w = 320
        base_pos = [0, 0, 0]
        cam_dist = 2
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)
    def dynamics(self,state,control):
          next_state = None
          dt = 0.05
          g = 9.81
          mc = 1
          mp = 0
          l = 0.5

          x_t, theta_t, dx_t, dtheta_t = np.split(state,4, axis=0)
          F = control

          ddtheta_t = (g*np.sin(theta_t)-np.cos(theta_t)  \
                    *(F+mp*l*dtheta_t**2*np.sin(theta_t))/(mc+mp))/  \
                    (l*(4/3-(mp*np.cos(theta_t)**2)/(mc+mp)))
          ddx_t = (F + mp*l*(dtheta_t**2*np.sin(theta_t) - ddtheta_t*np.cos(theta_t)))/(mc+mp)

          dx_t1 = dx_t + dt*ddx_t
          dtheta_t1 = dtheta_t + dt * ddtheta_t

          x_t1 = x_t + dt*dx_t1
          theta_t1 = theta_t + dt*dtheta_t1
          next_state = np.concatenate([x_t1, theta_t1, dx_t1, dtheta_t1], axis=0)
          
          return next_state.reshape((-1,1))


    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (4,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (4, 4) representing Jacobian df/dx for dynamics f
            B: np.array of shape (4, 1) representing Jacobian df/du for dynamics f
        """
        A, B = [], []
        # --- Your code here
        n = state.size
        m = control.size
        C = np.eye(n)
        D = np.eye(m)
        for i in range(n):
          state_plus_eps = self.dynamics(state + eps*C[i], control)
          state_minus_eps = self.dynamics(state - eps*C[i], control)

          df = (state_plus_eps - state_minus_eps)/(2*eps)
          A.append(df)

        for i in range(m):
          control_plus_eps = self.dynamics(state, control + eps*D[i])
          control_minus_eps = self.dynamics(state, control - eps*D[i])

          df = (control_plus_eps - control_minus_eps)/(2*eps)
          B.append(df)

        A = np.concatenate(A,axis = 1)
        B = np.concatenate(B,axis = 0)

        # ---
        return A, B


def dynamics_analytic(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 4) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 4) representing the next cartpole state

    """
    next_state = None
    dt = 0.05
    g = 9.81
    mc = 1
    mp = 0
    l = 0.5

    # --- Your code here

    x_t, theta_t, dx_t, dtheta_t = torch.chunk(input = state, chunks =4,dim=1)
    F = action

    ddtheta_t = (g*torch.sin(theta_t)-torch.cos(theta_t)  \
              *(F+mp*l*dtheta_t**2*torch.sin(theta_t))/(mc+mp))/  \
              (l*(4/3-(mp*torch.cos(theta_t)**2)/(mc+mp)))
    ddx_t = (F + mp*l*(dtheta_t**2*torch.sin(theta_t) - ddtheta_t*torch.cos(theta_t)))/(mc+mp)

    dx_t1 = dx_t + dt*ddx_t
    dtheta_t1 = dtheta_t + dt * ddtheta_t

    x_t1 = x_t + dt*dx_t1
    theta_t1 = theta_t + dt*dtheta_t1

    next_state = torch.cat([x_t1, theta_t1, dx_t1, dtheta_t1], dim=1)
    # ---

    return next_state


def linearize_pytorch(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (4,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

    """
    A, B = [], []
    # --- Your code here
    state = torch.unsqueeze(state,dim = 0)
    control = torch.unsqueeze(control,dim = 0)
    state.requires_grad = True
    control.requires_grad =True
    outputs = dynamics_analytic(state, control)
    outputs = torch.squeeze(outputs,dim=0)
    for i in range(outputs.shape[-1]):
      output_grad = torch.autograd.grad(outputs[i],(state,control),create_graph=True)
      A.append(output_grad[0])
      B.append(output_grad[1])
    print(A)
    A = torch.concatenate(A,dim = 0)
    B = torch.concatenate(B,dim=0)


    # ---
    return A, B
