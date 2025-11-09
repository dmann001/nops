"""
IMU Dead Reckoning Module
==========================

"""

import numpy as np
from collections import deque
import logging
import math
import time

# Initialize module logger
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def heading_deg_to_cardinal(heading_deg: float) -> str:
    """
    Convert heading in degrees to a cardinal string.
    
    Convention:
    - 0° = North (N)
    - 90° = East (E)
    - 180° = South (S)
    - 270° = West (W)
    
    This follows the standard compass convention where:
    - North is 0° (positive Y-axis in world frame)
    - East is 90° (positive X-axis in world frame)
    - Angles increase clockwise from North
    
    Note: The actual heading depends on:
    1. Device coordinate system (iOS/Android may differ)
    2. Device orientation (portrait/landscape)
    3. Magnetometer calibration
    """
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    # Normalize to [0, 360)
    h = heading_deg % 360.0
    idx = int((h + 22.5) // 45) % 8
    return dirs[idx]


class IMUDeadReckoningFixed:
    """
  

    Inputs per update:
      accel_x, accel_y, accel_z: linear accelerometer (m/s^2) including gravity
      gyro_x, gyro_y, gyro_z: angular rate (rad/s)
      timestamp_s: seconds (float)
      mag_x, mag_y, mag_z: optional magnetometer (uT)

    If your source timestamp is in ms, convert to seconds before calling update.
    """

    def __init__(self,
                 initial_position=(0.0, 0.0),
                 initial_heading_rad=0.0,
                 sample_rate_hz=50.0):
        """
        Initialize IMU dead reckoning.
        
        Args:
            initial_position: Starting position (x, y) in meters
            initial_heading_rad: Initial heading in radians (0 = North, π/2 = East)
            sample_rate_hz: Expected sample rate in Hz
        """
        # State
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)  # m/s in world frame
        self.heading = float(initial_heading_rad)          # radians (yaw)
        self.last_timestamp = None

        # Keep origin for optional snap-to-origin when back at start
        self.origin = np.array(initial_position, dtype=float)
        self.snap_to_origin_radius = 0.5  # meters (set None to disable snapping)

        # Buffers for stationary detection (use seconds->samples)
        self.sample_hz = float(sample_rate_hz)
        win_seconds = 0.5
        self.win_len = max(4, int(round(self.sample_hz * win_seconds)))
        self.accel_mag_buf = deque(maxlen=self.win_len)
        self.gyro_mag_buf = deque(maxlen=self.win_len)

        # Bias estimates (updated during stationary)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # Gravity estimate (low-pass on accelerometer when near gravity)
        self.gravity = np.array([0.0, 0.0, 9.81])
        self.gravity_alpha = 0.995  # slow adapt

        # High-pass filter memory for accel to remove residual low-freq components
        self.hp_state = np.zeros(3)
        self.hp_alpha = 0.92  # closer to 1 -> less high-pass (tunable: 0.85-0.98)

        # Magnetometer fusion
        self.mag_alpha = 0.25  # slightly higher trust in mag when moving

        # Stationary detection thresholds (empirically tuned for phone-in-pocket scenarios)
        self.accel_std_threshold = 0.18  # Standard deviation threshold for accelerometer (m/s²)
        self.gyro_std_threshold = 1.4 * (math.pi/180.0)  # Angular velocity threshold (~1.4°/s)
        self.stationary_required_windows = 2  # Consecutive windows needed to declare stationary

        # ReckonMe-inspired zero velocity detection thresholds
        # Based on: Renaudin et al. (2012), "Complete Triaxis Magnetometer Calibration"
        self.threshold_peaks_gravity = 0.40  # Gravity variation threshold for step detection
        self.threshold_peaks_user_acc = 0.40  # User acceleration peak threshold
        self.user_acc_threshold = 0.18  # Minimum user acceleration for motion detection
        self.user_gravity_threshold_x = 0.7  # Gravity X-component threshold
        self.user_gravity_threshold_y = 0.7  # Gravity Y-component threshold

        # Buffers for peak detection (store recent values to detect peaks)
        self.gravity_buf = deque(maxlen=self.win_len)      # Store gravity magnitude
        self.user_acc_buf = deque(maxlen=self.win_len)     # Store user acceleration magnitude
        self.gravity_x_buf = deque(maxlen=self.win_len)    # Store gravity X component
        self.gravity_y_buf = deque(maxlen=self.win_len)    # Store gravity Y component

        # Counters
        self.stationary_windows = 0
        self.is_stationary = False
        self.last_motion_time = None
        self.no_motion_timeout = 3.0  # seconds of no motion => force freeze

        # Integration safety parameters and velocity constraints
        self.velocity_damping = 0.75  # Damping factor to prevent unbounded drift (0-1)
        self.velocity_threshold = 0.025  # Minimum velocity cutoff (m/s) - removes noise
        self.max_speed = 4.0  # Maximum human walking speed constraint (m/s)

        # Diagnostics / history
        self.history = []

        logger.info("IMUDeadReckoningFixed initialized")
        logger.info(f"  sample_hz={self.sample_hz}, window_len={self.win_len}")
        logger.info(f"  accel_std_th={self.accel_std_threshold}, gyro_std_th={self.gyro_std_threshold:.4f} rad/s")
        logger.info(f"  mag_alpha={self.mag_alpha}, hp_alpha={self.hp_alpha}")
        logger.info(f"  ReckonMe thresholds: gravity_peak={self.threshold_peaks_gravity}, "
                    f"user_acc_peak={self.threshold_peaks_user_acc}, step_th={self.user_acc_threshold}")

    # ----------------------------
    # Utility helpers
    # ----------------------------
    @staticmethod
    def _vec_norm(v):
        return float(np.linalg.norm(v))

    @staticmethod
    def _clamp_speed(v, max_speed):
        speed = np.linalg.norm(v)
        if speed > max_speed:
            return v * (max_speed / speed)
        return v

    @staticmethod
    def _wrap_angle(angle):
        # normalize to [-pi, pi)
        return math.atan2(math.sin(angle), math.cos(angle))

    # ----------------------------
    # Core update function
    # ----------------------------
    def update(self, accel_x, accel_y, accel_z,
               gyro_x, gyro_y, gyro_z,
               timestamp_s,
               mag_x=None, mag_y=None, mag_z=None):
        """
        Call at high rate (e.g., 50 Hz). timestamp_s is seconds (float).
        Returns state dict.
        """
        # initialize timestamp
        if self.last_timestamp is None:
            self.last_timestamp = float(timestamp_s)
            self.last_motion_time = float(timestamp_s)
            # initialize gravity with first accel reading (best-effort)
            self.gravity = np.array([accel_x, accel_y, accel_z])
            return self.get_state()

        dt = float(timestamp_s) - self.last_timestamp
        if dt <= 0 or dt > 1.0:
            # skip bad dt but update last_timestamp so we don't get stuck
            self.last_timestamp = float(timestamp_s)
            return self.get_state()

        # Raw vectors
        raw_acc = np.array([accel_x, accel_y, accel_z], dtype=float)
        raw_gyro = np.array([gyro_x, gyro_y, gyro_z], dtype=float)

        # Remove biases
        acc_unbiased = raw_acc - self.accel_bias
        gyro_unbiased = raw_gyro - self.gyro_bias

        # Magnitudes for stationarity metrics (use accel magnitude including gravity)
        acc_mag = np.linalg.norm(acc_unbiased)
        gyro_mag = np.linalg.norm(gyro_unbiased)

        # Compute user acceleration (linear acceleration after removing gravity)
        linear_acc = acc_unbiased - self.gravity
        user_acc_mag = np.linalg.norm(linear_acc)

        # Gravity magnitude and components
        gravity_mag = np.linalg.norm(self.gravity)
        gravity_x = self.gravity[0]
        gravity_y = self.gravity[1]

        # Push to rolling buffers (we use magnitude std to detect motion)
        self.accel_mag_buf.append(acc_mag)
        self.gyro_mag_buf.append(gyro_mag)

        # Push to ReckonMe-style buffers
        self.gravity_buf.append(gravity_mag)
        self.user_acc_buf.append(user_acc_mag)
        self.gravity_x_buf.append(abs(gravity_x))
        self.gravity_y_buf.append(abs(gravity_y))

        # Compute windowed std
        if len(self.accel_mag_buf) >= max(3, int(self.win_len / 2)):
            a_std = float(np.std(np.array(self.accel_mag_buf)))
            g_std = float(np.std(np.array(self.gyro_mag_buf)))
        else:
            a_std = 0.0
            g_std = 0.0

        # ReckonMe-style peak detection for gravity and user acceleration
        gravity_has_peak = False
        user_acc_has_peak = False

        if len(self.gravity_buf) >= 3:
            gravity_arr = np.array(self.gravity_buf)
            # Detect if there's a significant peak (max - min > threshold)
            gravity_range = float(np.max(gravity_arr) - np.min(gravity_arr))
            gravity_has_peak = gravity_range > self.threshold_peaks_gravity

        if len(self.user_acc_buf) >= 3:
            user_acc_arr = np.array(self.user_acc_buf)
            # Detect if there's a significant peak (max - min > threshold)
            user_acc_range = float(np.max(user_acc_arr) - np.min(user_acc_arr))
            user_acc_has_peak = user_acc_range > self.threshold_peaks_user_acc

        # Check if user acceleration exceeds step threshold
        user_acc_exceeds_step = user_acc_mag > self.user_acc_threshold

        # Check gravity component thresholds
        gravity_x_exceeds = abs(gravity_x) > self.user_gravity_threshold_x
        gravity_y_exceeds = abs(gravity_y) > self.user_gravity_threshold_y

        # Enhanced stationary detection logic (combine original + ReckonMe thresholds)
        is_stationary_candidate = (
            (a_std < self.accel_std_threshold) and
            (g_std < self.gyro_std_threshold) and
            not gravity_has_peak and
            not user_acc_has_peak and
            not user_acc_exceeds_step and
            not gravity_x_exceeds and
            not gravity_y_exceeds
        )

        if is_stationary_candidate:
            self.stationary_windows += 1
        else:
            self.stationary_windows = 0

        prev_stationary = self.is_stationary
        self.is_stationary = (self.stationary_windows >= self.stationary_required_windows)

        # If stationary, perform ZUPT and bias updates
        if self.is_stationary:
            # Simple gyro bias update via exponential smoothing
            if len(self.gyro_mag_buf) > 0:
                smooth_alpha = 0.99
                self.gyro_bias = smooth_alpha * self.gyro_bias + (1.0 - smooth_alpha) * raw_gyro

            # Zero velocity update
            self.velocity[:] = 0.0

            # Slowly adapt gravity estimate toward current accel (helps if phone small tilt)
            self.gravity = self.gravity_alpha * self.gravity + (1.0 - self.gravity_alpha) * acc_unbiased

            # record last motion time as now (we're stationary now)
            self.last_motion_time = float(timestamp_s)
        else:
            # Not stationary -> integrate and update motion timestamps
            self.last_motion_time = float(timestamp_s)

            # Update gravity only if accel close to gravity magnitude and gyro small (to avoid corrupting gravity)
            if abs(acc_mag - 9.81) < 1.5 and gyro_mag < (5.0 * self.gyro_std_threshold):
                self.gravity = self.gravity_alpha * self.gravity + (1.0 - self.gravity_alpha) * acc_unbiased

            # Compute linear acceleration by removing gravity (in device frame approximation)
            linear_acc = acc_unbiased - self.gravity

            # High-pass filter to remove low-frequency residuals (helps drift)
            self.hp_state = self.hp_alpha * self.hp_state + (1.0 - self.hp_alpha) * linear_acc
            accel_hp = linear_acc - self.hp_state

            # Transform acceleration from device frame to world frame using 2D rotation matrix
            # Standard coordinate transformation: R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            ax_body = accel_hp[0]
            ay_body = accel_hp[1]
            cos_h = math.cos(self.heading)
            sin_h = math.sin(self.heading)
            ax_world = ax_body * cos_h - ay_body * sin_h
            ay_world = ax_body * sin_h + ay_body * cos_h
            accel_world_2d = np.array([ax_world, ay_world])

            # Apply deadzone to suppress sensor noise below threshold
            accel_world_2d[np.abs(accel_world_2d) < 0.008] = 0.0

            # Integrate velocity
            self.velocity += accel_world_2d * dt

            # Damping if small motion (scaled with dt)
            self.velocity *= (1.0 - (1.0 - self.velocity_damping) * dt * 10.0)

            # clamp speed
            self.velocity = self._clamp_speed(self.velocity, self.max_speed)

        # Heading update: integrate gyro z (yaw)
        self.heading += (gyro_unbiased[2] * dt)  # gyro in rad/s
        self.heading = self._wrap_angle(self.heading)

        # Magnetometer-assisted heading correction
        # Fuses magnetometer readings with gyroscope integration to reduce yaw drift
        if mag_x is not None and mag_y is not None and mag_z is not None:
            mag_vec = np.array([mag_x, mag_y, mag_z], dtype=float)
            # Calculate magnetic heading (assumes device is mostly upright)
            mag_heading = math.atan2(mag_vec[1], mag_vec[0])

            if self.is_stationary:
                # Hard reset: Magnetometer is trusted completely when stationary
                # This eliminates accumulated gyroscope drift
                self.heading = self._wrap_angle(mag_heading)
            else:
                # Complementary filter: Weighted fusion during motion
                # Reduces susceptibility to magnetic disturbances while moving
                self.heading = self._wrap_angle(
                    (1.0 - self.mag_alpha) * self.heading + self.mag_alpha * mag_heading
                )

        # If stationary for too long, force hard stop
        if (float(timestamp_s) - self.last_motion_time) > self.no_motion_timeout:
            self.velocity[:] = 0.0

        # Apply velocity cutoff
        speed = np.linalg.norm(self.velocity)
        if speed < self.velocity_threshold:
            self.velocity[:] = 0.0

        # Integrate position
        old_pos = self.position.copy()
        self.position += self.velocity * dt
        dist_step = np.linalg.norm(self.position - old_pos)

        # Optional: snap to origin when stationary and close to it
        if self.is_stationary and self.snap_to_origin_radius is not None:
            dist_to_origin = np.linalg.norm(self.position - self.origin)
            if dist_to_origin < self.snap_to_origin_radius:
                self.position[:] = self.origin

        # Keep history for debugging
        self.history.append({
            't': float(timestamp_s),
            'pos_x': float(self.position[0]),
            'pos_y': float(self.position[1]),
            'vel_x': float(self.velocity[0]),
            'vel_y': float(self.velocity[1]),
            'speed': float(np.linalg.norm(self.velocity)),
            'acc_mag': float(acc_mag),
            'gyro_mag': float(gyro_mag),
            'is_stationary': bool(self.is_stationary),
            'heading_deg': float(math.degrees(self.heading) % 360.0)
        })

        # update time
        self.last_timestamp = float(timestamp_s)
        return self.get_state()

    # ----------------------------
    # Accessors
    # ----------------------------
    def get_state(self):
        speed = float(np.linalg.norm(self.velocity))
        return {
            'position': {'x': float(self.position[0]), 'y': float(self.position[1])},
            'velocity': {'vx': float(self.velocity[0]), 'vy': float(self.velocity[1]), 'speed': speed},
            'heading_rad': float(self.heading),
            'heading_deg': float(math.degrees(self.heading) % 360.0),
            'is_stationary': bool(self.is_stationary),
            'last_timestamp': float(self.last_timestamp) if self.last_timestamp else None
        }

    def reset(self, pos=(0.0, 0.0), heading_rad=0.0):
        self.__init__(initial_position=pos, initial_heading_rad=heading_rad, sample_rate_hz=self.sample_hz)


# ----------------------------
# AdvancedTrailTracker - Wrapper for server.py compatibility
# ----------------------------
class AdvancedTrailTracker:
    """
    High-level trail tracking interface for indoor positioning.
    
    Provides a simplified API for server integration while leveraging
    the full 6-DOF IMU dead-reckoning capabilities underneath.
    
    This wrapper handles:
        - Simplified sensor data input (2D tracking focused)
        - Trail history management
        - Position and heading state access
        - Reset and calibration functions
    """

    def __init__(self, expected_hz=100.0):
        """
        Initialize the trail tracker.

        Args:
            expected_hz: Expected data rate in Hz (50-100 Hz recommended)
        """
        self.expected_hz = float(expected_hz)
        self.imu = IMUDeadReckoningFixed(sample_rate_hz=self.expected_hz)

        # Map old parameter names to new ones for compatibility
        self.accel_move_threshold = 0.1  # Not used in new implementation, kept for compatibility
        self.accel_deadzone = 0.01       # Not used in new implementation, kept for compatibility
        self.velocity_threshold = self.imu.velocity_threshold
        self.stationary_threshold = self.imu.accel_std_threshold  # Map to accel_std_threshold
        self.velocity_damping = self.imu.velocity_damping

        logger.info(f"AdvancedTrailTracker initialized for {self.expected_hz} Hz data")

    def update(self, accel_x, accel_y, accel_z, gyro_z, timestamp,
               mag_x=None, mag_y=None, mag_z=None):
        """
        Update the trail tracker with new sensor data.

        Note: This interface only requires gyro_z (yaw rate) for simplicity.
        The underlying IMUDeadReckoningFixed uses full 6-DOF, so we pass gyro_x=0, gyro_y=0.

        Args:
            accel_x, accel_y, accel_z: Accelerometer data (m/s^2)
            gyro_z: Yaw rate (rad/s) - only z-axis gyro needed for 2D tracking
            timestamp: Timestamp in seconds (float)
            mag_x, mag_y, mag_z: Optional magnetometer data (uT)
        """
        state = self.imu.update(
            accel_x, accel_y, accel_z,
            0.0, 0.0, gyro_z,
            timestamp,
            mag_x=mag_x, mag_y=mag_y, mag_z=mag_z
        )

        return state

    @property
    def position(self):
        """Get current position."""
        return self.imu.position

    @position.setter
    def position(self, value):
        """Set position (updates underlying IMU state)."""
        self.imu.position = np.array(value, dtype=float)

    @property
    def heading(self):
        """Get current heading in radians."""
        return self.imu.heading

    @heading.setter
    def heading(self, value):
        """Set heading in radians (updates underlying IMU state)."""
        self.imu.heading = float(value)

    def get_state(self):
        """Get current state of the tracker."""
        return self.imu.get_state()

    def get_trail_data(self):
        """
        Get trail data for visualization.
        Returns a dict with 'path' containing list of position points.
        """
        path = []
        if self.imu.history:
            for entry in self.imu.history:
                path.append({
                    'x': entry['pos_x'],
                    'y': entry['pos_y'],
                    't': entry['t'],
                    'speed': entry['speed'],
                    'heading': entry['heading_deg']
                })

        return {
            'path': path,
            'current_state': self.get_state(),
            'total_points': len(path)
        }

    def reset(self):
        """Reset the trail tracker to initial state."""
        self.imu.reset()
        self.position = self.imu.position
        self.heading = self.imu.heading


# ----------------------------
# Quaternion-based Dead Reckoning (Madgwick + ZUPT)
# ----------------------------
class IMUDeadReckoningQuat:
    """
    Dead-reckoning using quaternion orientation (Madgwick filter) + ZUPT.

    Inputs per update:
      accel_x, accel_y, accel_z: accelerometer (m/s^2, including gravity)
      gyro_x, gyro_y, gyro_z: gyroscope (rad/s)
      mag_x, mag_y, mag_z: optional magnetometer (uT)
      timestamp_s: time in seconds (float)
    """

    def __init__(self,
                 initial_position=(0.0, 0.0),
                 sample_rate_hz=50.0,
                 acc_std_stationary=0.1,
                 gyro_std_stationary_deg=1.0):
        try:
            from ahrs.filters import Madgwick
            self.madgwick = Madgwick()
        except ImportError:
            raise ImportError("ahrs library required. Install with: pip install ahrs")

        # State
        self.position = np.array(initial_position, dtype=float)  # world x,y
        self.velocity = np.zeros(3)  # world x,y,z
        self.last_timestamp = None

        # Quaternion (w,x,y,z) – Madgwick manages this internally, but we keep a view
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # Current yaw heading (radians)
        self.yaw = 0.0

        # Stationary detection buffers
        self.sample_hz = float(sample_rate_hz)
        win_seconds = 0.5
        self.win_len = max(4, int(round(self.sample_hz * win_seconds)))
        self.accel_mag_buf = deque(maxlen=self.win_len)
        self.gyro_mag_buf = deque(maxlen=self.win_len)

        self.accel_std_threshold = float(acc_std_stationary)            # m/s^2
        self.gyro_std_threshold = float(gyro_std_stationary_deg) * math.pi / 180.0  # rad/s
        self.stationary_required_windows = 2

        self.is_stationary = False
        self.stationary_windows = 0
        self.no_motion_timeout = 3.0
        self.last_motion_time = None

        # Velocity damping / cutoff
        self.velocity_damping = 0.7
        self.velocity_threshold = 0.03  # m/s
        self.max_speed = 4.0  # m/s

        # Diagnostics
        self.history = []

        logger.info("IMUDeadReckoningQuat initialized")
        logger.info(f"  sample_hz={self.sample_hz}, window_len={self.win_len}")
        logger.info(f"  accel_std_th={self.accel_std_threshold}, gyro_std_th={self.gyro_std_threshold:.4f} rad/s")

    @staticmethod
    def _quat_to_rot(q):
        """Quaternion (w,x,y,z) → 3x3 rotation matrix (world-from-body)."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
        ])

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _clamp_speed(v, max_speed):
        speed = np.linalg.norm(v)
        if speed > max_speed:
            return v * (max_speed / speed)
        return v

    def _update_stationary(self, acc_mag, gyro_mag):
        self.accel_mag_buf.append(acc_mag)
        self.gyro_mag_buf.append(gyro_mag)

        if len(self.accel_mag_buf) >= max(3, int(self.win_len / 2)):
            a_std = float(np.std(np.array(self.accel_mag_buf)))
            g_std = float(np.std(np.array(self.gyro_mag_buf)))
        else:
            a_std = 0.0
            g_std = 0.0

        candidate = (a_std < self.accel_std_threshold) and (g_std < self.gyro_std_threshold)
        if candidate:
            self.stationary_windows += 1
        else:
            self.stationary_windows = 0

        self.is_stationary = (self.stationary_windows >= self.stationary_required_windows)
        return a_std, g_std

    def update(self, accel_x, accel_y, accel_z,
               gyro_x, gyro_y, gyro_z,
               timestamp_s,
               mag_x=None, mag_y=None, mag_z=None):
        """
        Call at high rate (e.g., 50–100 Hz). Returns current state dict.
        """
        if self.last_timestamp is None:
            self.last_timestamp = float(timestamp_s)
            self.last_motion_time = float(timestamp_s)
            return self.get_state()

        dt = float(timestamp_s) - self.last_timestamp
        if dt <= 0 or dt > 1.0:
            self.last_timestamp = float(timestamp_s)
            return self.get_state()

        acc = np.array([accel_x, accel_y, accel_z], dtype=float)
        gyr = np.array([gyro_x, gyro_y, gyro_z], dtype=float)

        acc_mag = np.linalg.norm(acc)
        gyro_mag = np.linalg.norm(gyr)

        # --- Stationary detection (before orientation) ---
        a_std, g_std = self._update_stationary(acc_mag, gyro_mag)

        # --- Madgwick orientation update (quaternion) ---
        if mag_x is not None and mag_y is not None and mag_z is not None:
            mag = np.array([mag_x, mag_y, mag_z], dtype=float)
            self.madgwick.updateMARG(gyr=gyr, acc=acc, mag=mag)
        else:
            self.madgwick.updateIMU(gyr=gyr, acc=acc)

        self.q = np.array(self.madgwick.Q, dtype=float)

        # Rotation matrix body->world
        R = self._quat_to_rot(self.q)

        # --- Remove gravity in world frame ---
        # World-frame acceleration (includes gravity).
        world_acc = R @ acc
        # Subtract gravity along world +Z
        world_acc[2] -= 9.81

        # Zero-velocity update if stationary
        if self.is_stationary:
            self.velocity[:] = 0.0
            self.last_motion_time = float(timestamp_s)
        else:
            self.last_motion_time = float(timestamp_s)
            # Integrate velocity
            self.velocity += world_acc * dt
            # Damping & clamping
            self.velocity *= self.velocity_damping
            self.velocity = self._clamp_speed(self.velocity, self.max_speed)

        # Safety: if stationary too long, hard stop
        if (float(timestamp_s) - self.last_motion_time) > self.no_motion_timeout:
            self.velocity[:] = 0.0

        # Velocity deadzone
        speed = np.linalg.norm(self.velocity)
        if speed < self.velocity_threshold:
            self.velocity[:] = 0.0

        # Integrate position (world X/Y only)
        old_pos = self.position.copy()
        self.position += self.velocity[:2] * dt
        dist_step = np.linalg.norm(self.position - old_pos)

        # --- Heading / yaw from quaternion ---
        w, x, y, z = self.q
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.yaw = self._wrap_angle(yaw)
        
        heading_deg = float(math.degrees(self.yaw) % 360.0)
        heading_cardinal = heading_deg_to_cardinal(heading_deg)

        # Log
        self.history.append({
            't': float(timestamp_s),
            'pos_x': float(self.position[0]),
            'pos_y': float(self.position[1]),
            'vel_x': float(self.velocity[0]),
            'vel_y': float(self.velocity[1]),
            'speed': float(speed),
            'acc_mag': float(acc_mag),
            'gyro_mag': float(gyro_mag),
            'is_stationary': bool(self.is_stationary),
            'heading_deg': heading_deg,
            'heading_cardinal': heading_cardinal,
        })

        self.last_timestamp = float(timestamp_s)
        return self.get_state()

    def get_state(self):
        speed = float(np.linalg.norm(self.velocity))
        w, x, y, z = self.q
        heading_deg = float(math.degrees(self.yaw) % 360.0)
        return {
            'position': {'x': float(self.position[0]), 'y': float(self.position[1])},
            'velocity': {'vx': float(self.velocity[0]),
                         'vy': float(self.velocity[1]),
                         'vz': float(self.velocity[2]),
                         'speed': speed},
            'quaternion': {'w': float(w), 'x': float(x), 'y': float(y), 'z': float(z)},
            'heading_rad': float(self.yaw),
            'heading_deg': heading_deg,
            'heading_cardinal': heading_deg_to_cardinal(heading_deg),
            'is_stationary': bool(self.is_stationary),
            'last_timestamp': float(self.last_timestamp) if self.last_timestamp else None
        }

    def reset(self, pos=(0.0, 0.0)):
        self.position = np.array(pos, dtype=float)
        self.velocity[:] = 0.0
        self.last_timestamp = None
        self.accel_mag_buf.clear()
        self.gyro_mag_buf.clear()
        self.stationary_windows = 0
        self.is_stationary = False
        self.madgwick.Q = np.array([1.0, 0.0, 0.0, 0.0])
        self.q = np.array(self.madgwick.Q, dtype=float)
        self.yaw = 0.0
        self.history = []


# ----------------------------
# AdvancedTrailTrackerQuat - Wrapper for quaternion-based tracking
# ----------------------------
class AdvancedTrailTrackerQuat:
    """
    Same style as AdvancedTrailTracker, but using quaternion-based IMUDeadReckoningQuat.
    """

    def __init__(self, expected_hz=100.0):
        self.expected_hz = float(expected_hz)
        self.imu = IMUDeadReckoningQuat(sample_rate_hz=self.expected_hz)
        
        # Expose attributes that server.py expects
        self.accel_move_threshold = 0.1  # Not used in new implementation, kept for compatibility
        self.accel_deadzone = 0.01  # Not used in new implementation, kept for compatibility
        self.velocity_threshold = self.imu.velocity_threshold
        self.stationary_threshold = self.imu.accel_std_threshold  # Map to accel_std_threshold
        self.velocity_damping = self.imu.velocity_damping
        
        logger.info(f"AdvancedTrailTrackerQuat initialized for {self.expected_hz} Hz data")

    def update(self, accel_x, accel_y, accel_z, gyro_z, timestamp,
               mag_x=None, mag_y=None, mag_z=None):
        # If you only have gyro_z from iOS, set gx, gy = 0
        return self.imu.update(
            accel_x, accel_y, accel_z,
            0.0, 0.0, gyro_z,
            timestamp,
            mag_x=mag_x, mag_y=mag_y, mag_z=mag_z
        )

    def get_state(self):
        return self.imu.get_state()
    
    def get_trail_data(self):
        """
        Get trail data for visualization.
        Returns a dict with 'path' containing list of position points.
        """
        # Extract path from history
        path = []
        if self.imu.history:
            for entry in self.imu.history:
                path.append({
                    'x': entry['pos_x'],
                    'y': entry['pos_y'],
                    't': entry['t'],
                    'speed': entry['speed'],
                    'heading': entry['heading_deg'],
                    'heading_cardinal': entry.get('heading_cardinal')
                })
        
        return {
            'path': path,
            'current_state': self.get_state(),
            'total_points': len(path)
        }

    @property
    def position(self):
        return self.imu.position

    @position.setter
    def position(self, value):
        self.imu.position = np.array(value, dtype=float)
    
    @property
    def heading(self):
        """Get current heading in radians."""
        w, x, y, z = self.imu.q
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return float(yaw)
    
    @heading.setter
    def heading(self, value):
        """Set heading - resets quaternion to match yaw angle."""
        # This is a simplified setter - in practice, you'd need to reconstruct the full quaternion
        # For now, we'll just log a warning
        logger.warning("Setting heading directly on quaternion-based tracker is not fully supported")
    
    def reset(self):
        """Reset the trail tracker to initial state."""
        self.imu.reset()


# ----------------------------
# Standalone UDP Listener with Real-time Tracking
# ----------------------------
def run_madgwick_udp_listener(udp_port=8888, use_plot=True):
    """
    Standalone UDP listener for real-time IMU tracking with Madgwick filter.
    
    Expects UDP packets in CSV format: ax,ay,az,gx,gy,gz,mx,my,mz
    
    Args:
        udp_port: UDP port to listen on (default: 8888)
        use_plot: If True, shows real-time matplotlib plot (default: True)
    """
    try:
        from ahrs.filters import Madgwick
    except ImportError:
        raise ImportError("ahrs library required. Install with: pip install ahrs")
    
    if use_plot:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError:
            logger.warning("matplotlib not available, disabling plot")
            use_plot = False
    
    import socket
    import threading
    
    # ----------------------------
    # 1. UDP SETUP
    # ----------------------------
    UDP_IP = "0.0.0.0"
    UDP_PORT = udp_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"✅ Listening for IMU data on UDP port {UDP_PORT}...")
    logger.info(f"✅ Listening for IMU data on UDP port {UDP_PORT}...")
    
    # ----------------------------
    # 2. GLOBAL VARIABLES
    # ----------------------------
    latest_imu = None
    imu_lock = threading.Lock()
    
    # ----------------------------
    # 3. PROCESSING VARIABLES
    # ----------------------------
    madgwick = Madgwick()
    position = np.zeros(3)
    velocity = np.zeros(3)
    prev_time = None
    
    ACC_NOISE_THRESHOLD = 0.08
    GYRO_NOISE_THRESHOLD = 0.02
    VELOCITY_DAMPING = 0.8
    
    # ----------------------------
    # 4. UDP LISTENER THREAD
    # ----------------------------
    def udp_listener():
        global latest_imu
        while True:
            data, addr = sock.recvfrom(1024)
            try:
                # Expect CSV: ax,ay,az,gx,gy,gz,mx,my,mz
                ax, ay, az, gx, gy, gz, mx, my, mz = map(float, data.decode().split(","))
                imu_sample = {
                    'acc': np.array([ax, ay, az]),
                    'gyro': np.array([gx, gy, gz]),
                    'mag': np.array([mx, my, mz]),
                    'timestamp': time.time()
                }
                with imu_lock:
                    latest_imu = imu_sample
            except Exception as e:
                continue
    
    threading.Thread(target=udp_listener, daemon=True).start()
    
    # ----------------------------
    # 5. POSITION TRACKING FUNCTION
    # ----------------------------
    xs, ys = [], []
    line = None
    ax_plot = None
    
    def update(frame):
        global latest_imu, prev_time, position, velocity, line, ax_plot
        
        with imu_lock:
            imu = latest_imu
        
        if imu is None:
            return (line,) if line is not None else tuple()
        
        now = imu['timestamp']
        if prev_time is None:
            prev_time = now
            return (line,) if line is not None else tuple()
        
        dt = now - prev_time
        prev_time = now
        
        acc = imu['acc']
        gyr = imu['gyro']
        mag = imu['mag']
        
        # -------- MADGWICK SENSOR FUSION WITH MAGNETOMETER --------
        madgwick.updateMARG(gyr=gyr, acc=acc, mag=mag)
        q = madgwick.Q
        qw, qx, qy, qz = q
        
        # -------- EXTRACT PLANAR YAW --------
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        # -------- ROTATE HORIZONTAL ACCELERATION --------
        ax_h, ay_h = acc[0], acc[1]  # horizontal accelerations from phone frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        world_acc_xy = np.array([
            cos_yaw * ax_h - sin_yaw * ay_h,
            sin_yaw * ax_h + cos_yaw * ay_h
        ])
        
        # -------- ADD VERTICAL ACCELERATION (REMOVE GRAVITY) --------
        world_acc = np.array([world_acc_xy[0], world_acc_xy[1], acc[2] - 9.81])
        
        # -------- DRIFT SUPPRESSION --------
        if np.linalg.norm(world_acc_xy) < ACC_NOISE_THRESHOLD and np.linalg.norm(gyr) < GYRO_NOISE_THRESHOLD:
            world_acc_xy = np.zeros(2)
            velocity[:2] *= VELOCITY_DAMPING
        
        # -------- INTEGRATE HORIZONTAL VELOCITY --------
        velocity[:2] += world_acc_xy * dt
        velocity[:2] *= VELOCITY_DAMPING
        position[:2] += velocity[:2] * dt
        
        # Optional: keep Z only for reference
        velocity[2] += world_acc[2] * dt
        velocity[2] *= VELOCITY_DAMPING
        position[2] += velocity[2] * dt
        
        # -------- UPDATE PLOT --------
        if use_plot and line is not None:
            xs.append(position[0])
            ys.append(position[1])
            line.set_data(xs, ys)
            ax_plot.relim()
            ax_plot.autoscale_view()
        
        return (line,) if line is not None else tuple()
    
    # ----------------------------
    # 6. PLOT SETUP (if enabled)
    # ----------------------------
    if use_plot:
        plt.ion()
        fig, ax_plot = plt.subplots()
        ax_plot.set_xlabel("X Position (m)")
        ax_plot.set_ylabel("Y Position (m)")
        ax_plot.set_title("2D IMU Tracking (Magnetometer Corrected)")
        line, = ax_plot.plot([], [], "-o")
        
        # ----------------------------
        # 7. RUN ANIMATION
        # ----------------------------
        ani = FuncAnimation(fig, update, interval=50)
        plt.show()
        
        # Keep running until window is closed
        try:
            plt.pause(0.1)
            while plt.get_fignums():
                plt.pause(0.1)
        except KeyboardInterrupt:
            logger.info("Plot closed by user")
    else:
        # Without plot, just run the update loop
        logger.info("Running without plot. Press Ctrl+C to stop.")
        try:
            while True:
                update(0)
                time.sleep(0.02)  # ~50 Hz update rate
        except KeyboardInterrupt:
            logger.info("Stopped by user")
    
    sock.close()
    logger.info("UDP listener stopped")


# ----------------------------
# Main entry point
# ----------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "madgwick":
        # Run the standalone UDP listener
        port = 8888
        if len(sys.argv) > 2:
            try:
                port = int(sys.argv[2])
            except ValueError:
                logger.warning(f"Invalid port {sys.argv[2]}, using default 8888")
        
        use_plot = True
        if len(sys.argv) > 3 and sys.argv[3].lower() == "noplot":
            use_plot = False
        
        run_madgwick_udp_listener(udp_port=port, use_plot=use_plot)
    else:
        print("Usage:")
        print("  python imu_dead_reckoning.py madgwick [port] [noplot]")
        print("  Example: python imu_dead_reckoning.py madgwick 8888")
        print("  Example: python imu_dead_reckoning.py madgwick 8888 noplot")
