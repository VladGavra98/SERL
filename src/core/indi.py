""" Incremental Nonlinear Dynamic Inversion (INDI) attitude controller.

Sensor-based / incremental attitude-rate INDI for the fixed-wing PHLab task:

    u_cmd = u0 + B(+)(m_des - x0)

with
    x0    estimated angular acceleration  [p_dot, q_dot, r_dot]   (measurement)
    m_des desired angular acceleration from the attitude/rate loops
    B     control-effectiveness matrix  d[p_dot, q_dot, r_dot]/d[de, da, dr]
    u0    currently applied deflection  [de, da, dr]

The pseudo-control (angular acceleration) is estimated by a filtered
finite difference of the measured rates (sensor-based INDI). The control
effectiveness is initialised by a short in-sim probing manoeuvre and then
adapted online with three recursive least-squares estimators (one per axis).

Steinert et al., "From fundamentals to applications of incremental nonlinear dynamic inversion: A survey on INDI - Part I"
"""

import numpy as np

EPS = 1e-6 


class _AngularAccelEstimator:
    """ Filtered backward finite difference of the measured rates. """

    def __init__(self, dt: float, window: int = 5, cutoff_hz: float = 5.0):
        self.dt = float(dt)
        self.window = max(2, int(window))
        self.alpha = float(np.exp(-2.0 * np.pi * cutoff_hz * self.dt))
        self.reset()

    def reset(self) -> None:
        self._rates: np.ndarray = np.zeros((0, 3))
        self.current: np.ndarray = np.zeros(3)

    def update(self, omega: np.ndarray) -> np.ndarray:
        if self._rates.shape[0] >= self.window:
            self._rates = np.vstack((self._rates[1:], np.asarray(omega)))
        else:
            self._rates = np.vstack((self._rates, np.asarray(omega)))

        n = self._rates.shape[0]
        if n < 2:
            self.current = np.zeros(3)
            return self.current

        span = n - 1
        accel = (self._rates[-1] - self._rates[-1 - span]) / (span * self.dt)
        self.current = self.alpha * self.current + (1.0 - self.alpha) * accel
        return self.current


class _BmEstimator:
    """ Per-axis Recursive Least Squares estimates of the 3x3 control effectiveness.

    Each row i of B is estimated from  y_i = bias + b_i^T [de, da, dr],
    the bias absorbing the (unmodelled) state-dependent aerodynamic moments.
    """

    def __init__(self, lam: float = 0.9995, exc_threshold: float = 1e-4,
                 n_ctrl: int = 3):
        self.lam = float(lam)
        self.exc_threshold = max(10*EPS,float(exc_threshold))
        self.n_ctrl = n_ctrl
        self.reset()

    def reset(self) -> None:
        self.theta: np.ndarray = np.zeros((3, self.n_ctrl + 1))
        self.P: np.ndarray = np.stack([np.eye(self.n_ctrl + 1) * 100.0] * 3)

    def set_basis(self, B: np.ndarray, bias: np.ndarray = None) -> None:
        """ Initialise from an a-priori estimate (e.g. probing manoeuvre). """
        self.theta[:, 1:] = np.asarray(B, dtype=float)
        self.theta[:, 0] = np.zeros(3) if bias is None else np.asarray(bias)
        P_diag = np.ones(self.n_ctrl + 1)
        P_diag[0] = 1.0
        P_diag[1:] = EPS
        self.P = np.stack([np.diag(P_diag)] * 3)

    def update(self, u: np.ndarray, accel: np.ndarray, u_prev: np.ndarray) -> None:
        """ One RLS recursion once per control update. """
        if np.max(np.abs(np.asarray(u) - np.asarray(u_prev))) < self.exc_threshold:
            return
        phi = np.concatenate(([1.0], np.asarray(u, dtype=float)))
        for i in range(3):
            P = self.P[i]
            th = self.theta[i]
            Pphi = P @ phi
            gain = Pphi / (self.lam + phi @ Pphi)
            th = th + gain * (accel[i] - phi @ th)
            P = (P - np.outer(gain, Pphi)) / self.lam
            P = 0.5 * (P + P.T)
            self.P[i] = P
            self.theta[i] = th

    @property
    def B(self) -> np.ndarray:
        return self.theta[:, 1:]

    @property
    def bias(self) -> np.ndarray:
        return self.theta[:, 0]


class INDIController:
    """ Cascaded INDI attitude controller tracking [theta, phi, beta] refs. """

    def __init__(self, env, dt: float = 0.01,
                 K_theta: float = 1.5, K_phi: float = 0.6, K_beta: float = 0.7,
                 K_rate: float = 3.0, K_rate_yaw: float = 3.0,
                 coord_feedforward: bool = True,
                 accel_window: int = 5,
                 accel_cutoff_hz: float = 5.0, rls_lam: float = 0.9995,
                 exc_threshold: float = 1e-4,
                 probe_amp: float = 0.05, probe_duration: float = 0.03,
                 probe_settle: float = 0.1):
        self.dt = float(dt)
        self.K_theta = float(K_theta)
        self.K_phi = float(K_phi)
        self.K_beta = float(K_beta)
        self.K_rate = float(K_rate)
        if K_rate_yaw is None:
            K_rate_yaw = self.K_rate
        self.K_rate_yaw = float(K_rate_yaw)
        self.coord_feedforward = bool(coord_feedforward)

        self.probe_amp = float(probe_amp)
        self.probe_duration = float(probe_duration)
        self.probe_settle = float(probe_settle)
        self.n_probe = max(1, int(round(self.probe_duration / self.dt)))

        self.n_ctrl = int(env.action_space.shape[0])
        assert self.n_ctrl == 3, "INDI controller requires the 3-command PHLab set-up."
        self.bounds = np.asarray(env.action_space.high, dtype=float)

        self._accel = _AngularAccelEstimator(self.dt, accel_window, accel_cutoff_hz)
        self._bm = _BmEstimator(lam=rls_lam, exc_threshold=exc_threshold,
                                n_ctrl=self.n_ctrl)

        self.last_u: np.ndarray = np.zeros(self.n_ctrl)
        self.last_u_prev: np.ndarray = self.last_u.copy()

    def reset(self) -> None:
        """ Clear estimator state (call after env.reset). """
        self._accel.reset()
        self._bm.reset()
        self.last_u = np.zeros(self.n_ctrl)
        self.last_u_prev = self.last_u.copy()

    def identify(self, env) -> None:
        """ Probe the plant with short +/- surface deflections to initialise B.
        """
        self.reset()
        n_settle = max(1, int(round(self.probe_settle / self.dt)))
        amp = self.probe_amp
        z = np.zeros(self.n_ctrl)

        def settle() -> None:
            for _ in range(n_settle):
                env._sim_step(z)
            return env.x[0:3]

        def probe_col(j: int, sign: float) -> np.ndarray:
            rate0 = settle()
            u = z.copy()
            u[j] = sign * amp
            for _ in range(self.n_probe):
                env._sim_step(u)
            return (env.x[0:3] - rate0) / (self.n_probe * self.dt)

        B = np.zeros((3, self.n_ctrl))
        bias = np.zeros(3)
        for j in range(self.n_ctrl):
            acc_plus = probe_col(j, +1.0)
            acc_minus = probe_col(j, -1.0)
            B[:, j] = (acc_plus - acc_minus) / (2.0 * amp)
            bias += 0.5 * (acc_plus + acc_minus)
        bias /= self.n_ctrl

        settle()
        self._accel.update(env.x[0:3])
        self._bm.set_basis(B, bias=bias)
        self.last_u = z.copy()
        self.last_u_prev = self.last_u.copy()

    def _solve(self, delta_response: np.ndarray) -> np.ndarray:
        """ Control allocation: regularised pseudo-inverse of the 3x3 B. """
        B = self._bm.B
        cov = B @ B.T
        pseudo_inv_B = B.T @ np.linalg.inv(cov + EPS * np.eye(3))   # adds some factor for numerical stability
        return pseudo_inv_B @ delta_response

    def update(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """ Compute and return the clamped deflection command (radians).

        Args:
            x (np.ndarray): DASMAT state of length 12 (p, q, r, V, alpha, beta,
                            phi, theta, psi, he, xe, ye).
            ref (np.ndarray): Desired [theta, phi, beta] in radians.

        Returns:
            np.ndarray: [de, da, dr] deflection command (radians).
        """
        omega = np.asarray(x[0:3])
        theta, phi, beta = x[7], x[6], x[5]

        accel = self._accel.update(omega)
        self._bm.update(self.last_u, accel, self.last_u_prev)

        omega_c = np.array([
            self.K_phi * (ref[1] - phi),
            self.K_theta * (ref[0] - theta),
            -self.K_beta * beta,
        ])
        if self.coord_feedforward:
            # coordinated-turn yaw-rate bias (g/V * phi) to prevent sideslip
            omega_c[2] += (9.80665 / max(x[3], 1.0)) * phi
        m_des = np.array([
            self.K_rate * (omega_c[0] - omega[0]),
            self.K_rate * (omega_c[1] - omega[1]),
            self.K_rate_yaw * (omega_c[2] - omega[2]),
        ])

        delta_cmd = self.last_u + self._solve(m_des - accel)
        delta_cmd = np.clip(delta_cmd, -self.bounds, self.bounds)

        self.last_u_prev = self.last_u
        self.last_u = delta_cmd
        return delta_cmd
