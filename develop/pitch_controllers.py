#!/usr/bin/env python3
"""
Linear pitch-attitude feedback controllers for the reduced short-period plant.

Both controllers share the same interface:

    ctrl.reset()          # clear all internal state
    delta_e = ctrl(e)     # e = theta_ref - theta   ->  elevator deflection [rad]

Input is the pitch-attitude error e(t), output is the elevator deflection
delta_e(t), clamped to +/- sat (radians).

Discrete-time implementation (constant sample interval dt):

    PID (vanilla):
        u  = Kp*e + I + d
        I += Ki*e*dt                     # integral (Euler)
        d  = low-pass(filtered diff of e) # derivative on error, noise-filtered

    Antiwindup PID (back-calculation):
        identical, but while the command is saturated the difference
        (u_sat - u_cmd) is fed back into the integrator through the tracking
        time constant Tt, giving smooth de-saturation instead of integral windup.

Derivative is taken on the error through a first-order low-pass so that step
changes in the reference do not produce an impact spike, keeping the derivative
term well-behaved in a sampled loop.
"""

import numpy as np
import control as ct


class PIDController:
    """Common discrete-time PID machinery shared by the plain and anti-windup
    variants. Subclasses override the integrator update."""

    def __init__(self, Kp=2.0, Ki=1.5, Kd=0.6, dt=0.1, sat=np.deg2rad(25.0)):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.dt = float(dt)
        self.sat = abs(float(sat))
        self.reset()

    def reset(self) -> None:
        """Clear integrator and derivative-filter state."""
        self._integral = 0.0
        self._e_prev = 0.0
        self._d_prev = 0.0
        self._initialized = False

    def _derivative(self, e: float) -> float:
        """Low-pass filtered backward difference of the error (on-error,
        avoids a derivative kick during reference steps)."""
        dt = self.dt
        if not self._initialized:
            self._e_prev = e
            self._initialized = True
        raw = (e - self._e_prev) / dt
        # first-order low-pass filter, time constant tau = dt
        alpha = dt / (dt + dt)
        self._d_prev = alpha * raw + (1.0 - alpha) * self._d_prev
        self._e_prev = e
        return self._d_prev

    def _compute_cmd(self, e: float, derivative: float) -> float:
        """Unclamped control command from the current error and filtered
        derivative. Plain PID uses the stored running integral directly.

        Returns the un-wound command; callers clamp and (for the anti-windup
        variant) feed the saturation back into the integral update.
        """
        return self.Kp * e + self._integral + self.Kd * derivative

    def _integrate(self, e: float) -> None:
        """Vanilla integrator: accumulate the integral term."""
        self._integral += self.Ki * e * self.dt

    def __call__(self, e: float) -> float:
        derivative = self._derivative(e)
        u_cmd = self._compute_cmd(e, derivative)
        self._integrate(e)
        return float(np.clip(u_cmd, -self.sat, self.sat))


class AntiwindupPIDController(PIDController):
    """PID controller with back-calculation anti-windup.

    The unscaled command u_cmd is computed exactly as in the plain PID. If it
    exceeds the actuator limits, the clipped value u_sat is the signal actually
    applied to the plant. The difference (u_sat - u_cmd)/Tt is injected back
    into the integrator so that the wound-up state is driven back toward the
    saturation boundary, rather than left to accumulate unboundedly.
    """

    def __init__(self, Kp=2.0, Ki=1.5, Kd=0.6, dt=0.1, sat=np.deg2rad(25.0),
                 Tt=None):
        super().__init__(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, sat=sat)
        # Tracking time constant; default ties de-saturation speed to Ki.
        self.Tt = float(Tt) if Tt is not None else 1.0 / max(self.Ki, 1e-9)

    def _integrate(self, e: float, u_cmd: float, u_sat: float) -> None:
        """Back-calculation: normal integral plus anti-windup feedback term."""
        back_calc = (u_sat - u_cmd) / self.Tt
        self._integral += self.Ki * e * self.dt + self.dt * back_calc

    def __call__(self, e: float) -> float:
        derivative = self._derivative(e)
        u_cmd = self._compute_cmd(e, derivative)
        u_sat = float(np.clip(u_cmd, -self.sat, self.sat))
        self._integrate(e, u_cmd, u_sat)
        return u_sat


class LQRController:
    """Infinite-horizon Linear-Quadratic-Regulator pitch controller.

    Uses the same interface as the PID variants:

        ctrl.reset()          # clear integrator / state estimate
        delta_e = ctrl(e)     # e = theta_ref - theta   ->  elevator deflection

    The plant state z = [alpha, q, theta] is reconstructed from an internal,
    model-exact discrete-time propagation (matching the linear plant), augmented
    with an error integrator so theta tracks theta_ref.  Control law is
    u = -K [alpha, q, theta, int(e)], clamped to +/- sat.

    Gains are the LQR solution of the ZOH-discretized augmented system and are
    NOT tuned here.
    """

    def __init__(self, A, B, C, dt, Q=None, R=1.0, x0=None, sat=np.deg2rad(25.0)):
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        C = np.atleast_2d(np.asarray(C, dtype=float))
        self._n = A.shape[0]
        self._A = A
        self._B = B
        self._C = C
        self.dt = float(dt)
        self.sat = abs(float(sat))
        self.R = float(R)

        if Q is None:
            nz = self._n + 1
            q = np.ones(nz)
            q[-1] = 100.0
            q[self._n - 1] = 10.0
            Q = np.diag(q)
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))

        # Augment with an error-integrator state:  z = [alpha, q, theta, int(e)]
        nz = self._n + 1
        A_aug = np.zeros((nz, nz))
        A_aug[: self._n, : self._n] = A
        A_aug[self._n, self._n - 1] = -1.0        # d/dt int(e) = -theta
        B_aug = np.vstack([B, [[0.0]]])
        print(A_aug)

        dt = float(dt)
        Ad = ct.c2d(ct.ss(A_aug, B_aug, np.zeros((1, nz)), np.zeros((1, 1))), dt, method='zoh')
        out = ct.dlqr(Ad.A, Ad.B, self.Q, self.R)
        print(out[0])
        K = out[0]
        self.K = np.asarray(K, dtype=float).reshape(-1)

        self._x = np.zeros(self._n)
        self.x0 = np.zeros(self._n) if x0 is None else np.asarray(x0, dtype=float)
        self._e_int = 0.0

    def reset(self, x0=None) -> None:
        """Reset the internal state estimate and error integrator."""
        self._x = np.asarray(x0, dtype=float).copy() if x0 is not None else self.x0.copy()
        self._e_int = 0.0

    def _step_x(self, u: float) -> None:
        """Advance the internal (model-exact) state estimate by one dt (ZOH)."""
        dt = self.dt
        self._x = self._x + (self._A * dt) @ self._x + (self._B[:, 0] * u) * dt

    def __call__(self, e: float) -> float:
        z = np.concatenate([self._x, [self._e_int]])
        u_cmd = -self.K @ z
        u_sat = float(np.clip(u_cmd, -self.sat, self.sat))
        self._e_int += e * self.dt
        self._step_x(u_sat)
        return u_sat
