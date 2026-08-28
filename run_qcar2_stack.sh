#!/usr/bin/env bash
# QCar2 all-in-one launcher + interactive Resume/E-Stop control.
#
# Brings up the full driving stack (Cartographer localization, obstacle
# avoidance, depth e-stop, sound, V2V receiver, dashboard, path_mpc) and
# gives you a live menu to start/stop the car safely while it runs.
#
# Run this INTERACTIVELY (ssh in, then run it in your own terminal) — the
# menu needs a real keyboard, and E-Stop should always be one keypress
# away from your own hands, not something scripted remotely.
#
# Why E-Stop is a real stop, not a pause: notes.md documents that
# `/motion_enable false` does not reliably stop the car mid-drive (Issue 9,
# unresolved), and that killing qcar2_hardware with SIGKILL does NOT stop
# the motors either — only SIGINT does, because the motor-zeroing code
# lives in the ~QCar2() destructor, which SIGKILL bypasses entirely
# (Issue 17). So E-Stop here means: SIGINT qcar2_hardware, confirm it
# actually exited, and treat driving as over until you relaunch.

set -u
# TEST WORKSPACE (ros2_ws_sami) -- carries the localization-jump fix in
# path_mpc_node.py / path_utils.py. Map, pbstream and trajectory still come
# from ros2_ws_izhan on purpose, so this is an identical-input comparison
# against the known-good workspace; only the code differs. Logs are written
# into ros2_ws_sami so izhan's logs are never clobbered.
cd ~/ros2_ws_sami || { echo "ros2_ws_sami not found"; exit 1; }

# .bashrc already sourced ros2_ws_izhan; source sami ON TOP so its build of
# qcar_science_night_pkg wins the overlay. Without this the script would
# silently run izhan's unfixed code from this directory.
# set -u is on (top of script) and colcon's setup.bash dereferences
# COLCON_TRACE unset, which aborts under -u. .bashrc sources izhan without
# -u so it never hit this. Drop -u just for the source, then restore.
set +u
source /home/nvidia/ros2_ws_sami/install/setup.bash
set -u
echo "workspace: ros2_ws_sami (localization-jump fix active)"

LOGDIR=~/qcar2_run_logs
POSE_FILE=/home/nvidia/ros2_ws_sami/last_known_pose.txt
SEED_FILE=/home/nvidia/ros2_ws_sami/last_known_pose.seed
mkdir -p "$LOGDIR"
ts() { date +%H:%M:%S; }

# Track backgrounded PIDs for direct `ros2 run` nodes. qcar2_hardware is
# found separately (see below) since it's a grandchild of `ros2 launch`.
declare -A PIDS

# Every process the localization/hardware launch group can spawn
# (qcar2_cartographer_launch.py): hardware, IMU, camera, LiDAR driver,
# Cartographer, and robot_state_publisher. robot_state_publisher runs
# from a stock ROS path (/opt/ros/humble/lib/...), NOT under qcar2_nodes,
# so it needs its own pattern term — missing this originally left orphaned
# duplicates behind after every relaunch (found 2026-08-27).
LOCGROUP_PATTERN='qcar2_nodes/lib/qcar2_nodes|cartographer_node|cartographer_occupancy_grid_node|robot_state_publisher'

# --------------------------------------------------------------------
find_hardware_pid() {
    pgrep -f 'qcar2_nodes/qcar2_hardware' | head -1
}

find_camera_pid() {
    pgrep -f 'qcar2_nodes/rgbd' | head -1
}

# SIGINT, confirm exit, before any -9 sweep touches it. Found 2026-08-27:
# rgbd (RealSense) has the same "must exit cleanly or leaves the hardware
# wedged" property as qcar2_hardware, just for the USB device state
# instead of motor state. A -9'd rgbd left the color stream stuck at
# ~93% CPU with no visible ROS node after the next instance started —
# looked like a normal launch (clean log, no error) but never actually
# produced frames. Needed a full power cycle to clear; a plain restart
# after SIGKILL did not recover it.
stop_camera_gracefully() {
    local cam; cam=$(find_camera_pid)
    if [ -z "$cam" ]; then return 0; fi
    kill -2 "$cam" 2>/dev/null
    for _ in $(seq 1 10); do
        sleep 0.5
        kill -0 "$cam" 2>/dev/null || return 0
    done
    echo "  WARNING: rgbd (camera) did not exit within 5s of SIGINT."
}

rotate_log() {
    # Preserve the previous run's log instead of truncating it — losing a
    # crash's tail to a blind overwrite is exactly what made the first
    # qcar2_hardware death (2026-08-27) impossible to diagnose after a
    # relaunch.
    [ -f "$1" ] && mv "$1" "$1.prev"
}

cleanup_localization_group() {
    # Kill every straggler from a previous localization/hardware launch
    # BEFORE starting a new one. rgbd gets a graceful SIGINT first (see
    # stop_camera_gracefully) — -9 on it wedges the RealSense USB stream,
    # confirmed 2026-08-27 (needed a full power cycle to clear). Safe to
    # -9 everything else in the pattern: the destructor/cleanup hazard is
    # specific to qcar2_hardware (motors) and rgbd (USB device state).
    local n
    n=$(ps aux | grep -E "$LOCGROUP_PATTERN" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  cleaning up $n straggler process(es) from a previous launch..."
        stop_camera_gracefully
        pkill -9 -f "$LOCGROUP_PATTERN" 2>/dev/null
        pkill -9 -f 'ros2 launch qcar2_nodes' 2>/dev/null
        sleep 2
    fi
}

wait_for_hardware() {
    echo "  waiting for qcar2_hardware to come up..."
    for _ in $(seq 1 30); do
        local pid; pid=$(find_hardware_pid)
        if [ -n "$pid" ]; then
            echo "  qcar2_hardware up (PID $pid)"
            return 0
        fi
        sleep 1
    done
    echo "  WARNING: qcar2_hardware did not appear within 30s — check $LOGDIR/localization.log"
    return 1
}

wait_for_mpc_ready() {
    echo "  waiting for path_mpc: \"Localization stable. MPC enabled.\" ..."
    for _ in $(seq 1 40); do
        if grep -q "Localization stable" "$LOGDIR/path_mpc.log" 2>/dev/null; then
            echo "  MPC ready."
            return 0
        fi
        sleep 1
    done
    echo "  WARNING: MPC did not report ready within 40s — check $LOGDIR/path_mpc.log"
    return 1
}

check_duplicates() {
    echo "--- duplicate check ---"
    ps aux | grep -E "qcar2_hardware|lidar_overtake|depth_emergency|path_mpc|sound_node|v2v_receiver|v2v_dashboard|$LOCGROUP_PATTERN" \
        | grep -v grep | grep -vE 'bash -c|bin/ros2' \
        | awk '{print $NF}' | sed 's#.*/##' | sort | uniq -c \
        | awk '{ if ($1 > 1) print "  DUPLICATE: " $0; else print "  ok: " $0 }'
}

# --------------------------------------------------------------------
# Seed cartographer with the pose path_mpc last trusted, rather than making
# it globally relocalise. The lua config now rejects weak global matches
# (they were teleporting the car mid-drive between two look-alike stretches
# of track), so recovery after an e-stop + relaunch comes from this seed
# instead. Purely additive: on any failure the auto-started trajectory is
# left running and the stack behaves as it did before.
snapshot_pose() {
    if [ -f "$POSE_FILE" ]; then
        cp "$POSE_FILE" "$SEED_FILE"
        echo "[$(ts)] snapshotted pose for reseed: $(cat "$SEED_FILE")"
    else
        rm -f "$SEED_FILE"
        echo "[$(ts)] no pose to snapshot - relaunch will not be seeded."
    fi
}

seed_cartographer() {
    # DISABLED 2026-08-28: seed_cartographer.py's pose math produced a real
    # ~180deg yaw error on its first actual use (SetInitialTrajectoryPose's
    # pose is relative to trajectory 0's own internal frame, not
    # necessarily the map TF frame -- see qcar2-cartographer-pose-teleport
    # memory). Deleting the .seed file after the fact was not enough to
    # stay disabled -- snapshot_pose() below recreates it from POSE_FILE
    # on every relaunch regardless -- so this is disabled at the source
    # until the pose math is verified. Falls back to proven unseeded
    # global relocalization, same as before seeding existed.
    echo "[$(ts)] seeding disabled pending a pose-math fix - using global relocalization."
    return 0
}

# --------------------------------------------------------------------
launch_stack() {
    echo "[$(ts)] launching localization (Cartographer) ..."
    cleanup_localization_group     # in case a previous run's script died without a clean quit
    rotate_log "$LOGDIR/localization.log"
    setsid nohup ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
        > "$LOGDIR/localization.log" 2>&1 < /dev/null &
    disown
    wait_for_hardware

    echo "[$(ts)] launching lidar_overtake ..."
    setsid nohup ros2 run qcar_science_night_pkg lidar_overtake --ros-args -r __node:=lidar_overtake \
        > "$LOGDIR/lidar_overtake.log" 2>&1 < /dev/null &
    disown; PIDS[lidar_overtake]=$!

    echo "[$(ts)] launching depth_emergency_node ..."
    setsid nohup ros2 run qcar_science_night_pkg depth_emergency_node --ros-args -r __node:=depth_emergency_node \
        > "$LOGDIR/depth_emergency.log" 2>&1 < /dev/null &
    disown; PIDS[depth_emergency]=$!

    echo "[$(ts)] launching sound_node ..."
    setsid nohup ros2 run qcar_science_night_pkg sound_node --ros-args -r __node:=sound_node \
        > "$LOGDIR/sound_node.log" 2>&1 < /dev/null &
    disown; PIDS[sound_node]=$!

    echo "[$(ts)] launching v2v_receiver ..."
    setsid nohup ros2 run qcar_science_night_pkg v2v_receiver --ros-args \
        --params-file install/qcar_science_night_pkg/share/qcar_science_night_pkg/config/v2v_params.yaml \
        > "$LOGDIR/v2v_receiver.log" 2>&1 < /dev/null &
    disown; PIDS[v2v_receiver]=$!

    echo "[$(ts)] launching dashboard ..."
    setsid nohup python3 v2v_dashboard.py --role qcar2 --peer-host 192.168.0.100 \
        --trajectory ~/ros2_ws_izhan/track_run_cartographer_final.npy \
        > "$LOGDIR/dashboard.log" 2>&1 < /dev/null &
    disown; PIDS[dashboard]=$!

    echo "[$(ts)] launching path_mpc ..."
    setsid nohup ros2 run qcar_science_night_pkg path_mpc --ros-args -r __node:=path_mpc \
        > "$LOGDIR/path_mpc.log" 2>&1 < /dev/null &
    disown; PIDS[path_mpc]=$!

    wait_for_mpc_ready
    check_duplicates
    echo "[$(ts)] stack up. Nothing drives until you press 'r' (resume)."
}

# --------------------------------------------------------------------
do_resume() {
    local hw; hw=$(find_hardware_pid)
    if [ -z "$hw" ]; then
        echo "qcar2_hardware is not running — press 'l' to relaunch the stack first."
        return
    fi
    if ! grep -q "Localization stable" "$LOGDIR/path_mpc.log" 2>/dev/null; then
        echo "path_mpc has not reported 'Localization stable' yet — check $LOGDIR/path_mpc.log before resuming."
        return
    fi
    echo "[$(ts)] RESUME — publishing /motion_enable true"
    # 'pub --once' can fire before DDS discovery has matched the ephemeral
    # publisher to path_mpc'"'"'s subscriber, silently dropping the message
    # -- confirmed live 2026-08-28 (two real Resume presses did nothing;
    # a manual republish immediately started the car). Publish a few times
    # over ~1s instead of trusting a single --once; the message is
    # idempotent (just sets a bool) so repeating it is harmless.
    for _ in 1 2 3; do
        ros2 topic pub --once /motion_enable std_msgs/msg/Bool "{data: true}" >/dev/null 2>&1
        sleep 0.3
    done
    ros2 topic pub --once /mission_restart std_msgs/msg/Bool "{data: true}" >/dev/null 2>&1
    echo "  car should be driving now — watch the dashboard or path_mpc.log"
}

do_estop() {
    local hw; hw=$(find_hardware_pid)
    if [ -z "$hw" ]; then
        echo "qcar2_hardware is already not running."
        return
    fi
    echo "[$(ts)] E-STOP — sending SIGINT to qcar2_hardware (PID $hw)"
    kill -2 "$hw"
    for _ in $(seq 1 10); do
        sleep 0.5
        if ! kill -0 "$hw" 2>/dev/null; then
            echo "  CONFIRMED STOPPED — qcar2_hardware exited cleanly."
            echo "  Motion is now impossible until you relaunch ('l')."
            return
        fi
    done
    echo "  WARNING: qcar2_hardware did NOT exit within 5s of SIGINT."
    echo "  Do not assume the car is stopped. If it's still moving, cut physical power now."
}

do_relaunch() {
    local hw; hw=$(find_hardware_pid)
    if [ -n "$hw" ]; then
        echo "qcar2_hardware is still running (PID $hw) — E-Stop it first ('e') before relaunching."
        return
    fi
    echo "[$(ts)] relaunching localization + hardware ..."
    # Clear any stragglers from the previous launch group FIRST — a bare
    # relaunch on top of a still-partially-alive group is what produced
    # duplicate qcar2_imu (and unflagged robot_state_publisher duplicates)
    # on 2026-08-27: qcar2_hardware exiting does not bring down its
    # siblings (Cartographer, IMU, camera, robot_state_publisher) on its own.
    snapshot_pose
    cleanup_localization_group
    rotate_log "$LOGDIR/localization.log"      # fresh process + fresh redirect below — safe to rename
    # path_mpc itself is NOT restarted here — it keeps its original file
    # descriptor across a relaunch, so path_mpc.log must be truncated
    # in place (same inode), never renamed, or nothing would ever write
    # to a fresh path_mpc.log again and wait_for_mpc_ready would hang.
    : > "$LOGDIR/path_mpc.log"
    setsid nohup ros2 launch qcar2_nodes qcar2_cartographer_launch.py \
        > "$LOGDIR/localization.log" 2>&1 < /dev/null &
    disown
    wait_for_hardware
    seed_cartographer
    wait_for_mpc_ready
    check_duplicates
}

do_status() {
    echo "--- status @ $(ts) ---"
    local hw; hw=$(find_hardware_pid)
    if [ -n "$hw" ]; then echo "  qcar2_hardware: RUNNING (PID $hw)"; else echo "  qcar2_hardware: STOPPED"; fi
    for name in lidar_overtake depth_emergency sound_node v2v_receiver dashboard path_mpc; do
        local pid=${PIDS[$name]:-}
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "  $name: RUNNING (PID $pid)"
        else
            echo "  $name: STOPPED"
        fi
    done
    echo "  last path_mpc line: $(tail -1 "$LOGDIR/path_mpc.log" 2>/dev/null | grep -v 'TF unavailable')"
    check_duplicates
}

full_shutdown() {
    echo "[$(ts)] shutting down everything ..."
    local hw; hw=$(find_hardware_pid)
    if [ -n "$hw" ]; then
        echo "  SIGINT qcar2_hardware (PID $hw), waiting for clean exit..."
        kill -2 "$hw"
        for _ in $(seq 1 10); do
            sleep 0.5
            kill -0 "$hw" 2>/dev/null || break
        done
        kill -0 "$hw" 2>/dev/null && echo "  WARNING: qcar2_hardware still alive — cut power if the car is moving."
    fi
    stop_camera_gracefully
    pkill -9 -f "$LOCGROUP_PATTERN|ros2 launch qcar2_nodes|path_mpc|lidar_overtake|depth_emergency_node|sound_node|v2v_receiver|v2v_dashboard" 2>/dev/null
    echo "  done."
}
trap full_shutdown EXIT INT TERM

# --------------------------------------------------------------------
launch_stack

while true; do
    echo
    echo "========================================================"
    echo " [r] Resume (start driving)   [e] E-STOP (stop driving)"
    echo " [l] Relaunch hardware/localization (after an E-Stop)"
    echo " [s] Status   [q] Quit (full shutdown)"
    echo "========================================================"
    read -rn1 -p "> " cmd
    echo
    case "$cmd" in
        r|R) do_resume ;;
        e|E) do_estop ;;
        l|L) do_relaunch ;;
        s|S) do_status ;;
        q|Q) break ;;
        *) echo "unrecognized: $cmd" ;;
    esac
done

exit 0
