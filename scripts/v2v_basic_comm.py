#!/usr/bin/env python3
"""Simple V2V communication demo for QCar2 <-> Husarion coordination.

This node publishes a small JSON payload containing the vehicle ID, intent,
and priority. It also subscribes to the same topic so each robot can receive
and react to the other robot's state.

For now the communication is intentionally simple:
- a shared ROS2 topic carries lightweight status messages
- a second command topic can carry commands such as stop/go/yield
- the receiving robot maps those commands to a simple motion decision

Example usage:
  python3 scripts/v2v_basic_comm.py --vehicle-id qcar2 --priority 1
  python3 scripts/v2v_basic_comm.py --vehicle-id husarion --priority 2

While running, type one of the following in the terminal:
  stop   -> publish a stop command to the command topic
  go     -> publish a go command to the command topic
  quit   -> exit the node
"""

import argparse
import json
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class V2VSignalNode(Node):
    def __init__(
        self,
        vehicle_id: str,
        priority: int,
        topic: str,
        rate_hz: float,
        cmd_topic: str,
        go_speed: float,
        yield_speed: float,
    ) -> None:
        super().__init__(f"v2v_signal_{vehicle_id}")
        self.vehicle_id = vehicle_id
        self.priority = priority
        self.topic = topic
        self.rate_hz = rate_hz
        self.cmd_topic = cmd_topic
        self.go_speed = go_speed
        self.yield_speed = yield_speed
        self.partner_state = None
        self.decision = "proceed"

        self.pub = self.create_publisher(String, self.topic, 10)
        self.command_pub = self.create_publisher(String, self.cmd_topic, 10)
        self.sub = self.create_subscription(String, self.topic, self._on_message, 10)
        self.command_sub = self.create_subscription(String, self.cmd_topic, self._on_command, 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self._publish_signal)
        self.running = True
        self.emergency_stop = False

    def _on_message(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Could not decode message: {exc}")
            return

        if payload.get("vehicle_id") == self.vehicle_id:
            return

        self.partner_state = payload
        self.decision = self._choose_decision(payload)
        self.get_logger().info(
            f"Received from {payload.get('vehicle_id')}: {payload.get('intent')} "
            f"(priority={payload.get('priority')}, decision={self.decision})"
        )

    def _choose_decision(self, partner_payload: dict) -> str:
        if not partner_payload:
            return "proceed"
        if self.emergency_stop:
            return "yield"
        if partner_payload.get("decision") == "yield" or partner_payload.get("intent") == "yield":
            return "yield"
        if partner_payload.get("priority", 0) > self.priority:
            return "yield"
        return "proceed"

    def _on_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Could not decode command: {exc}")
            return

        target = payload.get("target_vehicle")
        if target not in (None, "", self.vehicle_id):
            return

        action = str(payload.get("action", "")).lower()
        if action == "stop":
            self.emergency_stop = True
            self.decision = "yield"
            self.get_logger().info("Received stop command; stopping vehicle")
        elif action == "go":
            self.emergency_stop = False
            self.decision = "proceed"
            self.get_logger().info("Received go command; proceeding")
        elif action == "yield":
            self.emergency_stop = False
            self.decision = "yield"
            self.get_logger().info("Received yield command")

    def publish_command(self, action: str) -> None:
        payload = {
            "vehicle_id": self.vehicle_id,
            "target_vehicle": self.vehicle_id,
            "action": action,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.command_pub.publish(msg)
        self.get_logger().info(f"Published command={action} to {self.cmd_topic}")

    def _publish_signal(self) -> None:
        payload = {
            "vehicle_id": self.vehicle_id,
            "timestamp": time.time(),
            "intent": "yield" if self.decision == "yield" else "go",
            "priority": self.priority,
            "state": "ready",
            "decision": self.decision,
            "topic": self.topic,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        self.get_logger().info(
            f"Decision={self.decision}; publishing status to {self.topic}"
        )


def stdin_listener(node: V2VSignalNode) -> None:
    while node.running:
        try:
            line = input().strip().lower()
        except EOFError:
            time.sleep(0.1)
            continue
        if not line:
            continue
        if line in {"q", "quit", "exit"}:
            node.running = False
            break
        if line in {"stop", "go", "yield"}:
            node.publish_command(line)
        else:
            print("Type stop, go, yield, or quit")


def one_shot_command(node: V2VSignalNode, action: str) -> None:
    if action in {"stop", "go", "yield"}:
        node.publish_command(action)
        node.get_logger().info(f"Sent one-shot command={action}")
    else:
        node.get_logger().warn(f"Unsupported one-shot action={action}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple V2V signal exchange")
    parser.add_argument("--vehicle-id", default="qcar2", help="Unique vehicle name")
    parser.add_argument("--priority", type=int, default=1, help="Higher priority yields")
    parser.add_argument("--topic", default="/v2v/intent", help="ROS topic used for exchange")
    parser.add_argument("--rate-hz", type=float, default=2.0, help="Publish rate in Hz")
    parser.add_argument("--cmd-topic", default="/v2v/commands", help="String command topic")
    parser.add_argument("--command", default="", help="One-shot command: stop, go, or yield")
    parser.add_argument("--go-speed", type=float, default=0.20, help="Speed to publish when proceeding")
    parser.add_argument("--yield-speed", type=float, default=0.0, help="Speed to publish when yielding")
    args = parser.parse_args()

    rclpy.init()
    node = V2VSignalNode(
        args.vehicle_id,
        args.priority,
        args.topic,
        args.rate_hz,
        args.cmd_topic,
        args.go_speed,
        args.yield_speed,
    )

    if args.command:
        one_shot_command(node, args.command)
        node.running = False
    else:
        thread = threading.Thread(target=stdin_listener, args=(node,), daemon=True)
        thread.start()

    try:
        if args.command:
            time.sleep(0.5)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
