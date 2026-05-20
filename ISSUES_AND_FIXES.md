# QCar2 Simulation — Issues & Fixes

A log of every significant problem hit during setup, in the order they occurred.

---

## 1. Official Quanser Package Uses ROS1 (catkin)

**Issue:** The Quanser `urdf_representations` repo ships a `CMakeLists.txt` that uses catkin (ROS1). Running `colcon build` fails immediately.

**Fix:** Replace the entire `CMakeLists.txt` with an `ament_cmake` version:

```cmake
cmake_minimum_required(VERSION 3.8)
project(qcar2)
find_package(ament_cmake REQUIRED)
install(DIRECTORY meshes urdf launch
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
```

---

## 2. `ros2 pkg prefix qcar2` Returns Empty

**Issue:** After step 1 the package still wasn't found because the previous failed catkin build left broken state in the workspace.

**Fix:** Clean the workspace (`rm -rf build/ install/ log/`) and rebuild with `colcon build --packages-select qcar2`.

---

## 3. Mesh URIs Not Resolved in Gazebo

**Issue:** Gazebo prints `[Err] ... No mesh found at model://qcar2/meshes/...` and the car renders as a box.

**Root cause:** `GZ_SIM_RESOURCE_PATH` was not set, so Gazebo could not find the `share/qcar2` directory that contains the meshes.

**Fix:** Export the path before launching Gazebo:
```bash
export GZ_SIM_RESOURCE_PATH=$(ros2 pkg prefix qcar2)/share:$GZ_SIM_RESOURCE_PATH
```

---

## 4. `robot_state_publisher` Fails with XML Parse Error

**Issue:** Passing the URDF as a shell argument produced:
```
Couldn't parse parameter override rule '...<robot>...'
```
The shell splits the XML string on spaces, breaking the argument parser.

**Fix:** Create a dedicated `rsp.launch.py` that opens the URDF file internally and passes it as a Python string:
```python
with open(urdf_file, 'r') as f:
    robot_description = f.read()
Node(package='robot_state_publisher', ...,
     parameters=[{'robot_description': robot_description}])
```

---

## 5. AckermannSteering Plugin Silently Not Loading

**Issue:** No error, but the plugin was never subscribing to `cmd_vel`. Publishing velocity commands had no effect.

**Root cause:** `GZ_SIM_SYSTEM_PLUGIN_PATH` was empty, so Gazebo could not find `libgz-sim8-ackermann-steering-system.so`.

**Fix:**
```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/opt/gz_sim_vendor/lib
```

The `.so` lives at `/opt/ros/jazzy/opt/gz_sim_vendor/lib/libgz-sim8-ackermann-steering-system.so`.

---

## 6. AckermannSteering Subscribed but Car Not Moving

**Issue:** After fixing the path, the plugin subscribed to the topic, but the log showed no "Found joints" line and the car stayed still.

**Root cause:** Wrong XML parameter names in the URDF. We had used:
```xml
<left_wheel_joint>, <right_wheel_joint>, <rear_left_wheel_joint>, <rear_right_wheel_joint>
```
These are **not valid** parameter names for the AckermannSteering plugin.

**Fix:** The correct names are `<left_joint>` and `<right_joint>`, which can be repeated for multi-wheel-per-side setups:
```xml
<left_joint>wheel_frontLeft_joint</left_joint>
<left_joint>wheel_rearLeft_joint</left_joint>
<right_joint>wheel_frontRight_joint</right_joint>
<right_joint>wheel_rearRight_joint</right_joint>
```

---

## 7. Car Drifting at an Angle Without Commands (DiffDrive)

**Issue:** When DiffDrive was tried as an alternative, the front hub joints were not controlled by the plugin and freely rotated under physics, causing the car to drift sideways.

**Fix:** Switch back to AckermannSteering, which explicitly controls the hub (steering) joints via `<left_steering_joint>` and `<right_steering_joint>`.

---

## 8. `cmd_vel` Not Bridged — Messages Sent but Car Not Moving

**Issue:** Publishing on `/model/qcar2/cmd_vel` from ROS had no effect because no bridge was running between ROS and Gazebo.

**Fix:** Run `ros_gz_bridge` with the correct direction specifier (`]` = ROS→Gz):
```bash
ros2 run ros_gz_bridge parameter_bridge \
  /model/qcar2/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
```

---

## 9. Commands from Stopped Sessions Persisting

**Issue:** Stopping a single combined launch file did not properly kill all child processes. Old `cmd_vel` commands from a previous terminal session continued to drive the car.

**Fix:** Use separate terminals for Gazebo, RSP/spawn, and control nodes. Each process can then be killed independently with Ctrl-C.

---

## 10. Front Camera Showing Car Body Instead of Forward View

**Issue:** `rqt_image_view` on `/qcar2/front_camera/image` showed a close-up of a car part, not the road ahead.

**Root cause:** The `csi_front_joint` had `rpy="-1.5708 0 -1.5708"` from the original Quanser physical-robot URDF. In gz-sim the camera renders along the link's +X axis; this rotation made +X point sideways into the car body.

**Fix:** Change the joint's RPY so that the camera's +X points forward and 15° downward (good for seeing the road ahead):
```xml
<origin rpy="0 0.2618 0" xyz="0.198 0 0.1095"/>
```
