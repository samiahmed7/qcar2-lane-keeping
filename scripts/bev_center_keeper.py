#!/usr/bin/env python3
"""Minimal BEV/HSV right-lane centre keeper — no classifier, no memory, no synthesis.

Pipeline, end to end, every frame:
    camera image
      -> HSV threshold            (white markings only)
      -> IPM bird's-eye warp      (lane lines become ~vertical, parallel)
      -> column histogram         (white pixels per BEV column)
      -> peak detection           (each lane line = one peak)
      -> pick centre-dash + right-edge, target = midpoint of the two
      -> P control: steer so the right-lane centre sits on the car centreline

Drives the car in the centre of the RIGHT lane (between the dashed centre line
and the solid right edge line).

    python3 scripts/bev_center_keeper.py
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# --- calibrated IPM (identical to bev_lane_detector_node) ---
IPM_SRC_RATIOS = [(0.003, 0.502), (0.336, 0.294), (0.666, 0.296), (0.994, 0.504)]
IPM_DST_RATIOS = [(0.200, 1.000), (0.200, 0.000), (0.800, 0.000), (0.800, 1.000)]

W, H = 640, 480
HSV_LO = np.array([0, 0, 180])      # white = any hue, low saturation, high value
HSV_HI = np.array([180, 80, 255])

# Lane geometry in BEV pixels. Measured from the sim: dashed centre line and the
# solid right edge sit ~394 px apart in the warp (= 0.5 m road half-width). Half
# of that is where the right-lane centre lives relative to a single detected line.
LANE_SPACING_PX = 394.0
LANE_HALF_PX = LANE_SPACING_PX / 2.0

SPEED = 0.30            # m/s forward
KP = 0.0020             # rad/s per pixel of lateral error
OMEGA_MAX = 0.6         # rad/s steering clamp
MIN_COL_PIXELS = 1500   # a column needs this many white px (summed) to count


class BevCenterKeeper(Node):
    def __init__(self):
        super().__init__('bev_center_keeper')
        src = np.float32([[rx * W, ry * H] for rx, ry in IPM_SRC_RATIOS])
        dst = np.float32([[rx * W, ry * H] for rx, ry in IPM_DST_RATIOS])
        self.M = cv2.getPerspectiveTransform(src, dst)

        self.pub = self.create_publisher(Twist, '/model/qcar2/cmd_vel', 10)
        self.dbg_pub = self.create_publisher(Image, '/qcar2/lane/simple_debug', 1)
        self.create_subscription(Image, '/qcar2/front_camera/image', self.on_img, 1)
        self.get_logger().info('BEV centre keeper ready — right lane, HSV+BEV, P control')

    # ---- pipeline ----
    def on_img(self, msg: Image):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        bgr = img if msg.encoding == 'bgr8' else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != W or bgr.shape[0] != H:
            bgr = cv2.resize(bgr, (W, H))

        # 1. HSV white mask
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LO, HSV_HI)

        # 2. bird's-eye warp
        bev = cv2.warpPerspective(mask, self.M, (W, H))

        # 3. column histogram over the bottom 60% (nearest road)
        hist = bev[int(H * 0.40):, :].sum(axis=0).astype(np.float32)
        kernel = np.ones(15, np.float32) / 15.0
        hs = np.convolve(hist, kernel, mode='same')

        # 4. peaks = local maxima above threshold, merged if within 40 px
        peaks = self._find_peaks(hs)

        # 5. classify: centre dash (peak left of car centreline), right edge (right of it)
        center = W / 2.0
        dash, redge = self._classify(peaks, hs, center)

        # 6. right-lane centre target
        if dash is not None and redge is not None:
            target = 0.5 * (dash + redge)
            mode = 'both'
        elif dash is not None:
            target = dash + LANE_HALF_PX
            mode = 'dash_only'
        elif redge is not None:
            target = redge - LANE_HALF_PX
            mode = 'edge_only'
        else:
            target = None
            mode = 'none'

        cmd = Twist()
        if target is not None:
            err = target - center                      # +ve: lane centre right of car
            omega = float(np.clip(-KP * err, -OMEGA_MAX, OMEGA_MAX))
            cmd.linear.x = SPEED
            cmd.angular.z = omega
        else:
            cmd.linear.x = SPEED * 0.35                 # crawl, keep heading
            err = float('nan')
        self.pub.publish(cmd)

        self.get_logger().info(
            f'[{mode:9s}] dash={dash} edge={redge} target='
            f'{("%.0f" % target) if target is not None else "--"} '
            f'err={err:+.1f}px omega={cmd.angular.z:+.3f}',
            throttle_duration_sec=0.4,
        )
        self._publish_debug(bev, dash, redge, target, center)

    @staticmethod
    def _find_peaks(hs):
        thr = max(hs.max() * 0.25, MIN_COL_PIXELS)
        peaks = []
        for x in range(2, W - 2):
            if hs[x] >= thr and hs[x] >= hs[x - 1] and hs[x] >= hs[x + 1]:
                if not peaks or x - peaks[-1] > 40:
                    peaks.append(x)
                elif hs[x] > hs[peaks[-1]]:
                    peaks[-1] = x
        return peaks

    @staticmethod
    def _classify(peaks, hs, center):
        if not peaks:
            return None, None
        left = [p for p in peaks if p < center]
        right = [p for p in peaks if p >= center]
        # centre dash = strongest peak left of the car centreline
        dash = max(left, key=lambda p: hs[p]) if left else None
        # right edge = nearest peak to the right of the car centreline
        redge = min(right) if right else None
        return dash, redge

    def _publish_debug(self, bev, dash, redge, target, center):
        vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        h = vis.shape[0]

        def vline(x, color, label):
            if x is None:
                return
            xi = int(round(x))
            cv2.line(vis, (xi, 0), (xi, h), color, 2)
            cv2.putText(vis, label, (xi - 30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        vline(center, (255, 255, 0), 'car')        # cyan  = car centreline
        vline(dash, (0, 255, 255), 'dash')         # yellow = detected centre dash
        vline(redge, (0, 255, 0), 'edge')          # green  = detected right edge
        vline(target, (255, 0, 255), 'target')     # magenta = right-lane centre

        out = Image()
        out.header.stamp = self.get_clock().now().to_msg()
        out.height, out.width = vis.shape[:2]
        out.encoding = 'bgr8'
        out.step = out.width * 3
        out.data = vis.tobytes()
        self.dbg_pub.publish(out)


def main():
    rclpy.init()
    node = BevCenterKeeper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
