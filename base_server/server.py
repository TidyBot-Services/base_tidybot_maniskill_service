"""Base bridge for ManiSkill — exposes mobile base control via RPC.

Protocol: multiprocessing.managers.BaseManager on port 50000 (same as MuJoCo version).
"""

import threading
import time
from multiprocessing.managers import BaseManager

import numpy as np

from .config import BASE_RPC_AUTHKEY, BASE_RPC_PORT


class SimBase:
    """Simulated base object exposed via RPC."""

    def __init__(self, server, env_idx=0):
        self._server = server
        self._env_idx = env_idx
        self._cmd_vel = [0.0, 0.0, 0.0]
        self._cmd_time = 0.0
        self._is_velocity_mode = False

        print("[base-bridge] Using world-frame poses (no local conversion)")

    def ensure_initialized(self):
        pass

    def get_full_state(self):
        state = self._server.get_state(env_idx=self._env_idx)
        return {
            "base_pose": np.array([state.base_x, state.base_y, state.base_theta]),
            "base_velocity": np.array([state.base_vx, state.base_vy, state.base_wz]),
        }

    def execute_action(self, target):
        pose = target["base_pose"]
        self._is_velocity_mode = False
        self._cmd_vel = [0.0, 0.0, 0.0]
        self._server.set_base_action([float(pose[0]), float(pose[1]), float(pose[2])], env_idx=self._env_idx)

    def set_target_velocity(self, vel, frame="global"):
        self._cmd_vel = [float(vel[0]), float(vel[1]), float(vel[2])]
        self._cmd_time = time.time()
        self._is_velocity_mode = True

    def stop(self):
        self._is_velocity_mode = False
        self._cmd_vel = [0.0, 0.0, 0.0]

    def reset(self):
        pass

    def get_battery_voltage(self):
        return 25.2

    def get_command_state(self):
        return {
            "is_velocity_mode": self._is_velocity_mode,
            "cmd_vel": list(self._cmd_vel),
            "cmd_time": self._cmd_time,
        }


_sim_base_instances = {}  # env_idx → SimBase


class _SimBaseProxy:
    def __init__(self, real):
        self._real = real

    def ensure_initialized(self):
        return self._real.ensure_initialized()

    def get_full_state(self):
        return self._real.get_full_state()

    def execute_action(self, target):
        return self._real.execute_action(target)

    def set_target_velocity(self, vel, frame="global"):
        return self._real.set_target_velocity(vel, frame=frame)

    def stop(self):
        return self._real.stop()

    def reset(self):
        return self._real.reset()

    def get_battery_voltage(self):
        return self._real.get_battery_voltage()

    def get_command_state(self):
        return self._real.get_command_state()


class _BaseBridgeManager(BaseManager):
    pass


class BaseBridge:
    """Protocol bridge: multiprocessing.managers RPC on port 50000."""

    def __init__(self, server, port=BASE_RPC_PORT, authkey=BASE_RPC_AUTHKEY, env_idx=0):
        self._server = server
        self._env_idx = env_idx
        self._port = port
        self._authkey = authkey
        self._thread = None
        self._manager = None
        self._running = False

    def start(self):
        global _sim_base_instances
        _sim_base_instances[self._env_idx] = SimBase(self._server, env_idx=self._env_idx)

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="base-bridge")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._manager is not None:
            try:
                server = self._manager.get_server()
                server.stop_event = True
            except Exception:
                pass
            self._manager = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        print("[base-bridge] Stopped")

    def _run(self):
        env_idx = self._env_idx

        def _factory():
            return _SimBaseProxy(_sim_base_instances[env_idx])

        # Each env gets a unique manager class to avoid registration conflicts
        manager_cls = type(f"_BaseBridgeManager_{env_idx}", (BaseManager,), {})
        manager_cls.register("Base", callable=_factory)
        self._manager = manager_cls(
            address=("0.0.0.0", self._port),
            authkey=self._authkey,
        )
        server = self._manager.get_server()
        print(f"[base-bridge] RPC server listening on port {self._port} (env {env_idx})")
        try:
            server.serve_forever()
        except Exception as e:
            if self._running:
                print(f"[base-bridge] Server error: {e}")
