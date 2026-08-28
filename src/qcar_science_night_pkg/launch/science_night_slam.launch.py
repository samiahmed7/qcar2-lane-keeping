import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    qcar2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("qcar2_nodes"),
                "launch",
                "qcar2_launch.py",
            )
        )
    )

    scan_fixer = Node(
        package="qcar_science_night_pkg",
        executable="scan_fixer",
        name="scan_fixer",
        output="screen",
    )

    ekf_node = Node(
        package="qcar2_nodes",
        executable="ekf_fusor.py",
        name="ekf_fusor",
        output="screen",
        parameters=[{"tf_pub": True}],
    )

    # Initialize EKF at fixed start position
    init_ekf_pose = TimerAction(
        period=5.0,   # wait for hardware to start
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "topic", "pub", "--once",
                    "/init_pose",
                    "geometry_msgs/msg/PoseWithCovarianceStamped",
                    "{header: {frame_id: 'odom'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}",
                ],
                output="screen",
            )
        ],
    )

    # Map server — serves the saved .pgm map
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "yaml_filename": "/home/nvidia/ros2_ws_izhan/track_map_new.yaml",
            "use_sim_time": False,
        }],
    )

    # Lifecycle manager to activate map_server and amcl
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
    )

    # AMCL localization
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            "/home/nvidia/ros2_ws_sami/src/qcar_science_night_pkg/config/amcl_config.yaml"
        ],
    )

    lidar_overtake = Node(
        package="qcar_science_night_pkg",
        executable="lidar_overtake",
        name="lidar_overtake",
        output="screen",
    )

    lane_centering = Node(
        package="qcar_science_night_pkg",
        executable="lane_centering_node",
        name="lane_centering",
        output="screen",
    )

    depth_emergency_node = Node(
        package="qcar_science_night_pkg",
        executable="depth_emergency_node",
        name="depth_emergency_node",
        output="screen",
    )

    sound_node = Node(
        package="qcar_science_night_pkg",
        executable="sound_node",
        name="sound_node",
        output="screen",
    )

    # MPC starts last — after AMCL has localized
    path_mpc = TimerAction(
        period=18.0,
        actions=[
            Node(
                package="qcar_science_night_pkg",
                executable="path_mpc",
                name="path_mpc",
                output="screen",
            )
        ],
    )


    return LaunchDescription([
        qcar2_launch,
        ekf_node,
        init_ekf_pose,
        map_server,
        lifecycle_manager,
        amcl,
        lidar_overtake,
        lane_centering,
        #depth_emergency_node,
        sound_node,
        #path_mpc,
    ])