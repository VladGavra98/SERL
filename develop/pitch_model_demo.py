#!/usr/bin/env python3
"""
Reduced longitudinal (short-period) pitch model of a fixed-wing aircraft.

This script builds the linear state-space model for the aircraft pitch dynamics
from explicit, physical aerodynamic stability/control coefficients, prints the
state-space representation and the resulting elevator-to-pitch transfer
function, and plots the open-loop pitch response.

Dependencies:  pip install control numpy scipy matplotlib
Run:           python3 test/pitch_model_demo.py

================================================================================
Model derivation (short-period approximation)
================================================================================
The symmetric (longitudinal) small-perturbation equations of motion, restricted
to the fast short-period subset and the pitch-attitude kinematic, are written in
terms of the non-dimensional coefficients.  States and input:

    x = [alpha, q, theta]^T        (angle of attack, pitch rate, pitch attitude)
    u = delta_e                     (elevator deflection)

    Working assumption: CZ_q ~ CZ_deltae ~ 0, so the Z-force enters only
    through CZ_alpha and alpha-dot is dominated by the angle-of-attack term.

                        .
    alpha = Z_alpha * alpha  +  q                                    (1)

                        .
    q     = M_alpha * alpha + M_q * q + M_deltae * delta_e           (2)

                        .
    theta = q                                                        (3)

where the dimensional derivatives are recovered from non-dimensional
coefficients as:

                                qbar * S
    Z_alpha   =   CZ_alpha  *  ----------          [1/s]
                                m * U1

                                qbar * S * cbar
    M_alpha   =   Cm_alpha  *  ---------------    [1/s^2]
                                       Iyy

                                qbar * S * cbar^2
    M_q       =   Cm_q     *  ------------------  [1/s]
                                2 * Iyy * U1

                                qbar * S * cbar
    M_deltae  =   Cm_deltae  *  ---------------   [1/s^2]
                                       Iyy

with dynamic pressure  qbar = 0.5 * rho * U1^2.

Assembling as x_dot = A*x + B*u,  y = C*x + D*u,  y = theta:

        | Z_alpha   1    0 |            |    0    |
    A = | M_alpha  M_q   0 |     B  =   | M_deltae|
        |    0      1    0 |            |    0    |

    C = [ 0  0  1 ]        D = [ 0 ]

The pitch-attitude stiffness is carried by Cm_alpha (there is no separate
"Cm_theta": theta dynamics enter only through the kinematic theta_dot = q).
================================================================================
"""

import os, sys
import numpy as np
import control as ct
import matplotlib.pyplot as plt

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, os.pardir))
for _p in (_BASE_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import envs   # gym-like learning env
import envs.config
import signals  
print("=" * 72)
print("REDUCED LONGITUDINAL (SHORT-PERIOD) PITCH MODEL")
print("=" * 72)
# ----------------------------------------------------------------------------
#  Aircraft parameters / flight condition
# ----------------------------------------------------------------------------
# Geometry & inertia of a small fixed-wing UAV
S      = 0.33    # wing planform area        [m^2]
cbar   = 0.16    # mean aerodynamic chord     [m]
m      = 2.5     # total mass                 [kg]
Iyy    = 0.05    # pitch moment of inertia    [kg m^2]

# Flight condition
rho_air = 1.225  # air density                [kg/m^3]
U1      = 25.0   # trim (true air-)speed      [m/s]
qbar    = 0.5 * rho_air * U1**2               # dynamic pressure [N/m^2]

# Non-dimensional stability / control coefficients (short-period subset)
CZ_alpha   = -0.989     # lift-curve slope-ish contribution to Z-force
Cm_alpha   = -0.02474   # pitch stiffness (alpha due to alpha)
Cm_q       = -1.545     # pitch damping
Cm_deltae  =  0.04948   # elevator control effectiveness

# ----------------------------------------------------------------------------
# Dimensional stability derivatives (see derivation above)
# ----------------------------------------------------------------------------
Z_alpha  = CZ_alpha   * (qbar * S) / (m * U1)            # [1/s]
M_alpha  = Cm_alpha   * (qbar * S * cbar) / Iyy           # [1/s^2]
M_q      = Cm_q       * (qbar * S * cbar**2) / (2 * Iyy * U1)   # [1/s]
M_deltae = Cm_deltae  * (qbar * S * cbar) / Iyy           # [1/s^2]

# ----------------------------------------------------------------------------
# 3Assemble the linear state-space model  x_dot = A x + B u , y = C x
# ----------------------------------------------------------------------------
A = np.array([
    [Z_alpha,   1.0,   0.0],
    [M_alpha,   M_q,   0.0],
    [0.0,       1.0,   0.0],
])

B = np.array([
    [0.0],
    [M_deltae],
    [0.0],
])

C = np.array([[0.0, 0.0, 1.0]])   # measure pitch attitude theta
D = np.array([[0.0]])

sys_ss = ct.ss(A, B, C, D)
sys_tf = ct.ss2tf(A, B, C, D)      # G(s) = theta(s) / delta_e(s)
sys_tf = ct.minreal(sys_tf)

# Zero out any numerical-noise leading terms so the printed transfer function
# reads cleanly (the 2nd-order numerator coefficient comes out ~1e-15).
num, den = ct.tfdata(sys_tf)
num = np.where(np.abs(num) < 1e-10, 0.0, num)
sys_tf = ct.tf(num, den)
print('poles: ', poles:=ct.poles(sys_tf))
periods = [2 * np.pi/abs(pole.imag + 10**(-9)) for pole in poles]
print(f'T:{periods} s')

Tavg = np.min(periods)
# ----------------------------------------------------------------------------
# Print out the model
# ----------------------------------------------------------------------------

print("\nAircraft / flight parameters")
print(f"  S      = {S:8.3f} m^2        rho   = {rho_air:.3f} kg/m^3")
print(f"  cbar   = {cbar:8.3f} m        U1    = {U1:.1f} m/s")
print(f"  m      = {m:8.3f} kg         qbar  = {qbar:.1f} N/m^2")
print(f"  Iyy    = {Iyy:8.4f} kg m^2")

print("\nStability/control coefficients (non-dimensional)")
print(f"  CZ_alpha  = {CZ_alpha: .4f}")
print(f"  Cm_alpha  = {Cm_alpha: .5f}")
print(f"  Cm_q      = {Cm_q: .4f}")
print(f"  Cm_deltae = {Cm_deltae: .5f}")

print("\nDimensional stability derivatives")
print(f"  Z_alpha  = {Z_alpha: .4f} [1/s]")
print(f"  M_alpha  = {M_alpha: .4f} [1/s^2]")
print(f"  M_q      = {M_q: .4f}     [1/s]")
print(f"  M_deltae = {M_deltae: .4f} [1/s^2]")

print("\nState-space model  (state x = [alpha, q, theta],  input u = delta_e)")
print("A =")
for row in A:
    print("   [ " + "  ".join(f"{v: 8.4f}" for v in row) + " ]")
print("B =")
for row in B:
    print("   [ " + "  ".join(f"{v: 8.4f}" for v in row) + " ]")
print("C = [ " + "  ".join(f"{v:7.4f}" for v in C[0]) + " ]")

print("\nTransfer function  G(s) = theta(s) / delta_e(s)")
print(f"G(s) = {sys_tf}")

dt = 0.02
Tfinal = 25.0
t = np.linspace(0.0, Tfinal, int(Tfinal//dt))

ref_theta_max = np.deg2rad(30)
amp_times = [2,4,6,15, 18]
amps = [1,10,30,-10, 0.2] 
theta_ref = np.deg2rad(np.array([[signals.SmoothedStepSequence(amp_times, amps, smooth_width=Tfinal//10)(_t)] for _t in t]))
# ----------------------------------------------------------------------------
#     Controllers 
# ----------------------------------------------------------------------------

delta_e_step = np.deg2rad(0.0)                # 5 deg elevator step command
limit =  np.deg2rad(3.0)
u = np.full_like(t, delta_e_step)
u = np.deg2rad(np.asarray([[signals.CosineSmoothedStep(t_start = 2, t_end = 5)(_t)] for _t in t]))


# ----------------------------------------------------------------------------
# Closed-loop simulation  (discrete-time Euler, feedback through the controller)
# ----------------------------------------------------------------------------
from pitch_controllers import PIDController, AntiwindupPIDController, LQRController
Ak = A * dt
Bk = B * dt

x0 = np.array([0.012, 0.0, 0.012])

def run_closed_loop(ctrl):
    """Run the plant with `ctrl(theta_ref - theta)` as the elevator command."""
    print(f"Simulate for {Tfinal}s with dt={dt}s")
    x = x0.copy()
    theta_sim = np.zeros(len(t))
    u_log = np.zeros(len(t))
    for k in range(len(t)):
        theta_k = x[2]
        e_k = theta_ref[k, 0] - theta_k

        # apply actuator limits
        u_log[k] = np.clip(ctrl(e_k), -limit, limit)

        if k < len(t) - 1:
            x = x + (Ak @ x) + (Bk[:, 0] * u_log[k])
        theta_sim[k] = x[2]
    return theta_sim, u_log

Kp, Ki, Kd = 0.4, 0.5, 0.1
x_sim_theta, u_sim = run_closed_loop(PIDController(Kp,Ki,Kd, dt = dt))
x_aw_theta,  u_aw  = run_closed_loop(AntiwindupPIDController(Kp,Ki,Kd, dt =dt, sat=limit))
x_lqr_theta, u_lqr = run_closed_loop(LQRController(A, B, C, dt=dt, x0=x0, sat=limit))

# ----------------------------------------------------------------------------
#  Plot  (closed-loop theta tracking + elevator commands)
# ----------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "pitch_response.png")

fig, ax = plt.subplots(1,2)
ax[0].plot(t, np.rad2deg(theta_ref), linewidth=1.6, color='green', linestyle='--', label="ref")
ax[0].plot(t, np.rad2deg(x_sim_theta), linewidth=1.6, color='blue', label="PID")
ax[0].plot(t, np.rad2deg(x_aw_theta), linewidth=1.6, color='red', label="Anti-windup PID")
ax[0].plot(t, np.rad2deg(x_lqr_theta), linewidth=1.6, color='purple', label="LQR")
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("$\\theta$ [deg]")
ax[0].grid(True, alpha=0.3)
ax[0].set_xlim(0, Tfinal)
ax[0].legend()

ax[1].plot(t, np.rad2deg(u_sim), linewidth=1.6, color='blue', label="PID")
ax[1].plot(t, np.rad2deg(u_aw), linewidth=1.6, color='red', label="Anti-windup PID")
ax[1].plot(t, np.rad2deg(u_lqr), linewidth=1.6, color='purple', label="LQR")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Elevator $\\delta_e$ [deg]")
ax[1].grid(True, alpha=0.3)
ax[1].set_xlim(0, Tfinal)
ax[1].legend()

fig.tight_layout()
fig.savefig(out_path, dpi = 150)
print(f"\nPitch-response figure saved to: {out_path}")


