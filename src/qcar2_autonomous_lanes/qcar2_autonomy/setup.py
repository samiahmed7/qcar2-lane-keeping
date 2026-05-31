from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'qcar2_autonomy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='QCar2 autonomous lane driving nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = qcar2_autonomy.perception_node:main',
            'validation_node = qcar2_autonomy.validation_node:main',
            'kinematics_node = qcar2_autonomy.kinematics_node:main',
            'pid_lane_follower_node = qcar2_autonomy.pid_lane_follower_node:main',
            'ai_lane_keeper_node = qcar2_autonomy.ai_lane_keeper_node:main',
            'state_machine_node = qcar2_autonomy.state_machine_node:main',
            'dwa_local_planner_node = qcar2_autonomy.dwa_local_planner_node:main',
            'lane_change_planner_node = qcar2_autonomy.lane_change_planner_node:main',
            'state_to_behavior_bridge_node = qcar2_autonomy.state_to_behavior_bridge_node:main',
            'cmd_vel_mux_node = qcar2_autonomy.cmd_vel_mux_node:main',
            'dl_lane_segmentation_node = qcar2_autonomy.dl_lane_segmentation_node:main',
            'yolo_lane_node = qcar2_autonomy.yolo_lane_node:main',
            'rfdetr_onnx_lane_node = qcar2_autonomy.rfdetr_onnx_lane_node:main',
            'gazebo_auto_tuner = qcar2_autonomy.gazebo_auto_tuner:main',
            'bev_lane_detector = qcar2_autonomy.bev_lane_detector_node:main',
            'bev_lane_detector_node = qcar2_autonomy.bev_lane_detector_node:main',
            'lane_controller = qcar2_autonomy.lane_controller_node:main',
            'lidar_obstacle = qcar2_autonomy.lidar_obstacle_node:main',
            'behavior_fsm = qcar2_autonomy.behavior_fsm_node:main',
            'safety_supervisor = qcar2_autonomy.safety_supervisor_node:main',
            'metrics_logger = qcar2_autonomy.metrics_logger_node:main',
            'placeholder_node = qcar2_autonomy.placeholder_node:main',
        ],
    },
)
