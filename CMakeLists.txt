cmake_minimum_required(VERSION 3.0.2)
project(qn_optimizer)

# Find packages.
find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

# Generate catkin cmake files for package.
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES qn_optimizer
  DEPENDS EIGEN3
)

# Add include directories.
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# BUILD
# Set up optimizer library.
add_library(${PROJECT_NAME} src/qn_optimizer.cpp)
# Set up example executables.
# example_simple
add_executable(example_simple examples/simple.cpp)
target_link_libraries(example_simple ${PROJECT_NAME})
# example_constrained
add_executable(example_constrained examples/constrained.cpp)
target_link_libraries(example_constrained ${PROJECT_NAME})

# INSTALL
# Install library.
install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
# Install public header files.
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
# Install examples.
install(TARGETS example_simple example_constrained
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)