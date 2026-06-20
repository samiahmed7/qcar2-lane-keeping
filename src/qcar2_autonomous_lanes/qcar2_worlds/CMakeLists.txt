cmake_minimum_required(VERSION 3.8)
project(qcar2_worlds)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  maps
  models
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
