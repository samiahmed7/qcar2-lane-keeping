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
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='QCar2 autonomous lane driving nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            # ML perception (debug overlay / path recording aid)
            'rfdetr_onnx_lane_node = qcar2_autonomy.rfdetr_onnx_lane_node:main',
            # BEV lane detector (classical white-line, right-lane) -- segmentation debug
            'bev_lane_detector_node = qcar2_autonomy.bev_lane_detector_node:main',
            # Path recording
            'record_path_node = qcar2_autonomy.record_path_node:main',
            # MPC stack: perception -> planner -> tracker (+ run logger)
            'mpc_lidar_obstacle_node = qcar2_autonomy.mpc_lidar_obstacle_node:main',
            'mpc_sensor_noise_node = qcar2_autonomy.mpc_sensor_noise_node:main',
            'mpc_reference_planner_node = qcar2_autonomy.mpc_reference_planner_node:main',
            'mpc_drive_node = qcar2_autonomy.mpc_drive_node:main',
            'mpc_logger_node = qcar2_autonomy.mpc_logger_node:main',
        ],
    },
)
