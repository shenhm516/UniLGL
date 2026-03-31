cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

#find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(TBB REQUIRED)

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "pybind11 include: ${pybind11_INCLUDE_DIR}")

include_directories(
    include
    src
    src/pcd_utils
    ${EIGEN3_INCLUDE_DIRS}
#    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# pybind11_add_module(
#     _core
#     src/pcd_utils/pcd_utils.cpp
# )

python_add_library(_core MODULE
    src/pcd_utils/pcd_utils.cpp
    src/pcd_utils/solve_gnc.cpp
    src/pcd_utils/pcd_conversions.cpp
    src/pcd_utils/filters.cpp
    WITH_SOABI
)

target_link_libraries(_core PRIVATE
#    ${PCL_LIBRARIES}
    pybind11::headers
    ${OpenCV_LIBS}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    tbb
)

target_compile_definitions(_core PRIVATE
    EIGEN_USE_BLAS
)

install(TARGETS _core DESTINATION .)