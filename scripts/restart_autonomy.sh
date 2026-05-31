#!/usr/bin/env bash
# Restart ONLY the autonomy nodes (keep Gazebo + bridge running), so code/param
# changes take effect without rebooting the sim. Run as a file so pkill -f does
# not match this shell's own command line.
for pat in 'rfdetr_onnx_lane_node' 'pid_lane_follower_node' \
           'lane_change_planner_node' 'cmd_vel_mux_node' 'state_machine_node'; do
    pkill -9 -f "$pat" 2>/dev/null
done
sleep 2
echo "old autonomy nodes killed; relaunching..."
exec "$(dirname "$0")/run_autonomy.sh"
