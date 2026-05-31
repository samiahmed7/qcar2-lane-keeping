cmake_minimum_required(VERSION 3.8)
project(qcar2_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  meshes
  rviz
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
