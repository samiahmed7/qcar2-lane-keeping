cmake_minimum_required(VERSION 3.8)
project(qcar2_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
