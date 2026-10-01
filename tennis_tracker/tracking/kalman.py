"""2-D constant-acceleration Kalman filter with Rauch-Tung-Striebel smoothing.

State: [x, y, vx, vy, ax, ay] in pixels / frame. Measurement: [x, y].
The acceleration states let the filter follow the ball's parabolic flight
and bounce transitions better than a constant-velocity model.
"""

from __future__ import annotations

import numpy as np


class ConstantAccelerationKalman:
    def __init__(self, dt: float = 1.0, process_noise: float = 2.0, measurement_noise: float = 4.0,
                 init_pos_std: float = 5.0, init_vel_std: float = 30.0, init_acc_std: float = 5.0):
        self.dt = dt
        d, d2 = dt, 0.5 * dt * dt
        self.F = np.array([[1, 0, d, 0, d2, 0],
                           [0, 1, 0, d, 0, d2],
                           [0, 0, 1, 0, d, 0],
                           [0, 0, 0, 1, 0, d],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]], dtype=np.float64)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = self.H[1, 1] = 1.0
        # jerk-driven process noise (discrete white-noise acceleration change)
        q = process_noise ** 2
        g = np.array([d2 * d / 3, d2, d])  # rough integration weights for x, v, a
        G = np.zeros((6, 2))
        G[0, 0] = G[1, 1] = g[0]
        G[2, 0] = G[3, 1] = g[1]
        G[4, 0] = G[5, 1] = g[2]
        self.Q = G @ G.T * q
        self.R = np.eye(2) * measurement_noise ** 2
        self.P0 = np.diag([init_pos_std ** 2] * 2 + [init_vel_std ** 2] * 2 + [init_acc_std ** 2] * 2)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        # histories for RTS smoothing
        self.x_pred: list[np.ndarray] = []
        self.P_pred: list[np.ndarray] = []
        self.x_upd: list[np.ndarray] = []
        self.P_upd: list[np.ndarray] = []

    # ----------------------------------------------------------------- basics
    @property
    def initialised(self) -> bool:
        return self.x is not None

    def init(self, x: float, y: float) -> None:
        self.x = np.array([x, y, 0, 0, 0, 0], dtype=np.float64)
        self.P = self.P0.copy()
        self.x_pred, self.P_pred = [self.x.copy()], [self.P.copy()]
        self.x_upd, self.P_upd = [self.x.copy()], [self.P.copy()]

    def predict(self) -> np.ndarray:
        assert self.x is not None and self.P is not None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x_pred.append(self.x.copy())
        self.P_pred.append(self.P.copy())
        # provisional "update" = prediction, overwritten if update() follows
        self.x_upd.append(self.x.copy())
        self.P_upd.append(self.P.copy())
        return self.x[:2].copy()

    def innovation(self, z) -> tuple[np.ndarray, np.ndarray]:
        """Residual y and its covariance S for measurement z (after predict)."""
        assert self.x is not None and self.P is not None
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis2(self, z) -> float:
        y, S = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z) -> np.ndarray:
        assert self.x is not None and self.P is not None
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form (numerically safe)
        self.x_upd[-1] = self.x.copy()
        self.P_upd[-1] = self.P.copy()
        return self.x[:2].copy()

    @property
    def position(self) -> np.ndarray:
        assert self.x is not None
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        assert self.x is not None
        return self.x[2:4].copy()

    # -------------------------------------------------------------- smoothing
    def rts_smooth(self) -> np.ndarray:
        """Rauch-Tung-Striebel backward pass over the stored history.

        Returns smoothed states (T, 6). Requires >= 1 stored step.
        """
        n = len(self.x_upd)
        if n == 0:
            return np.zeros((0, 6))
        xs = [None] * n
        Ps = [None] * n
        xs[-1] = self.x_upd[-1].copy()
        Ps[-1] = self.P_upd[-1].copy()
        for k in range(n - 2, -1, -1):
            P_pred_next = self.P_pred[k + 1]
            try:
                C = self.P_upd[k] @ self.F.T @ np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                C = self.P_upd[k] @ self.F.T @ np.linalg.pinv(P_pred_next)
            xs[k] = self.x_upd[k] + C @ (xs[k + 1] - self.x_pred[k + 1])
            Ps[k] = self.P_upd[k] + C @ (Ps[k + 1] - P_pred_next) @ C.T
        return np.stack(xs)
