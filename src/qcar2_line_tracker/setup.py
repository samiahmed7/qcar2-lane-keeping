from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'qcar2_line_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Simple lane follower for QCar2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lane_follower = qcar2_line_tracker.lane_follower:main',
            'ipm_calibrate = qcar2_line_tracker.ipm_calibrate:main',
            'lane_filter   = qcar2_line_tracker.lane_filter:main',
        ],
    },
)
