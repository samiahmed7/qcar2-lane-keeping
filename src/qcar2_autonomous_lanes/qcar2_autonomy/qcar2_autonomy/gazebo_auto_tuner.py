#!/usr/bin/env python3
"""Automated Gazebo SIL tuning for vision EMA and steering PID parameters."""
import itertools
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageCaptureNode(Node):
    """Temporary image subscriber used by the tuner during one simulation run."""

    def __init__(self, image_topic, output_dir, capture_period_sec):
        super().__init__('gazebo_auto_tuner_capture_node')
        self.bridge = CvBridge()
        self.output_dir = Path(output_dir)
        self.capture_period_sec = float(capture_period_sec)
        self.last_capture_time = 0.0
        self.frame_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.create_subscription(Image, image_topic, self._on_image, 10)

    def _on_image(self, msg: Image):
        now = time.monotonic()
        if now - self.last_capture_time < self.capture_period_sec:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warn(
                f'cv_bridge conversion failed while capturing image: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        filename = self.output_dir / f'frame_{self.frame_count:05d}.png'
        if cv2.imwrite(str(filename), bgr):
            self.frame_count += 1
            self.last_capture_time = now


class GazeboAutoTuner:
    """Grid-search orchestration for simulation-in-the-loop parameter tuning."""

    def __init__(self):
        self.alpha_grid = np.linspace(0.01, 0.10, 10)
        self.kp_grid = [0.8, 1.0, 1.2]
        self.ki_grid = [0.0, 0.01]
        self.kd_grid = [0.0, 0.05, 0.10]

        self.image_topic = '/camera/image_raw'
        self.image_width = 640
        self.image_height = 480
        self.test_duration_sec = 30.0
        self.capture_period_sec = 0.5
        self.startup_wait_sec = 8.0
        self.output_root = Path('/tmp/qcar2_auto_tuner')
        self.results_csv = self.output_root / 'tuning_results.csv'

    def run(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        rows = []

        for alpha, kp, ki, kd in itertools.product(
            self.alpha_grid,
            self.kp_grid,
            self.ki_grid,
            self.kd_grid,
        ):
            run_dir = self.output_root / self._run_name(alpha, kp, ki, kd)
            processes = []
            rmse = math.inf
            try:
                processes = self._launch_simulation(alpha, kp, ki, kd)
                time.sleep(self.startup_wait_sec)
                self._capture_images(run_dir)
                rmse = self._rmse_from_images(run_dir)
            finally:
                self._teardown(processes)

            row = {
                'alpha': float(alpha),
                'Kp': float(kp),
                'Ki': float(ki),
                'Kd': float(kd),
                'rmse_px': float(rmse),
            }
            rows.append(row)
            self._append_result(row)

        return pd.DataFrame(rows)

    def _launch_simulation(self, alpha, kp, ki, kd):
        """Start Gazebo and ROS nodes for one parameter combination."""
        commands = [
            ['ros2', 'launch', 'qcar2_bringup', 'sim_bringup.launch.py'],
            [
                'ros2',
                'run',
                'qcar2_autonomy',
                'perception_node',
                '--ros-args',
                '-p',
                f'image_topic:={self.image_topic}',
            ],
            [
                'ros2',
                'run',
                'qcar2_autonomy',
                'validation_node',
                '--ros-args',
                '-p',
                f'alpha:={float(alpha)}',
            ],
            [
                'ros2',
                'run',
                'qcar2_autonomy',
                'lane_controller',
                '--ros-args',
                '-p',
                f'kp:={float(kp)}',
                '-p',
                f'ki:={float(ki)}',
                '-p',
                f'kd:={float(kd)}',
            ],
        ]

        env = os.environ.copy()
        return [
            subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            for command in commands
        ]

    def _capture_images(self, output_dir):
        """Spin a temporary subscriber and save snapshots for one run."""
        rclpy.init(args=None)
        node = ImageCaptureNode(
            image_topic=self.image_topic,
            output_dir=output_dir,
            capture_period_sec=self.capture_period_sec,
        )
        try:
            end_time = time.monotonic() + self.test_duration_sec
            while time.monotonic() < end_time and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def _rmse_from_images(self, image_dir):
        cte_values = []
        for image_path in sorted(Path(image_dir).glob('*.png')):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            lane_center = self._detect_lane_center_x(image)
            if lane_center is None:
                continue
            cte_values.append(lane_center - (self.image_width / 2.0))

        if not cte_values:
            return math.inf

        errors = np.array(cte_values, dtype=np.float32)
        return float(np.sqrt(np.mean(errors * errors)))

    def _detect_lane_center_x(self, bgr):
        """Estimate lane center from one saved camera image."""
        resized = cv2.resize(
            bgr,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Histogram projection: each x bin counts bright pixels in the road
        # half of the image. The left/right maxima approximate lane boundaries;
        # their midpoint is the lane center for CTE scoring.
        histogram = np.sum(binary[binary.shape[0] // 2:, :] > 0, axis=0)
        midpoint = histogram.shape[0] // 2
        left_x = int(np.argmax(histogram[:midpoint]))
        right_x = int(np.argmax(histogram[midpoint:]) + midpoint)

        if histogram[left_x] == 0 or histogram[right_x] == 0:
            return None
        return 0.5 * (left_x + right_x)

    def _teardown(self, processes):
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)

    def _append_result(self, row):
        result = pd.DataFrame([row])
        header = not self.results_csv.exists()
        result.to_csv(self.results_csv, mode='a', index=False, header=header)

    @staticmethod
    def _run_name(alpha, kp, ki, kd):
        return f'alpha_{alpha:.3f}_kp_{kp:.3f}_ki_{ki:.3f}_kd_{kd:.3f}'


def main():
    tuner = GazeboAutoTuner()
    tuner.run()


if __name__ == '__main__':
    main()
