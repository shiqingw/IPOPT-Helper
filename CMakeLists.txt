cmake_minimum_required(VERSION 3.18..3.20)
project(IpoptHelper)

# Set the build type to Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address,undefined")

# Set the build type to Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(OpenBLAS REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-python REQUIRED)

# Detect the operating system and set the library extensions and paths accordingly
# Set the Ipopt include directories
set(IPOPT_INCLUDE_DIRS /usr/local/include/coin-or)
set(IPOPT_LIBRARY_DIRS /usr/local/lib)
if(APPLE)
    set(LIB_SUFFIX ".dylib")
else()
    set(LIB_SUFFIX ".so")
endif()
# Manually specify the Ipopt libraries
set(IPOPT_LIBRARIES 
${IPOPT_LIBRARY_DIRS}/libipopt${LIB_SUFFIX}
${IPOPT_LIBRARY_DIRS}/libcoinmumps${LIB_SUFFIX}
)

pybind11_add_module(
    IpoptHelper
    src/pybind.cpp
    src/ellipsoidAndLogSumExp.cpp
)

target_link_libraries(IpoptHelper PUBLIC
    pybind11::module
    xtensor-python
    Python::NumPy
    ${OpenBLAS_LIBRARIES}  # Use the variable set by find_package if available
    ${IPOPT_LIBRARIES}
    pthread
    m
    dl
)

# Include directories
target_include_directories(IpoptHelper PUBLIC 
    ${IPOPT_INCLUDE_DIRS}
)

# Link Ipopt libraries
target_link_libraries(IpoptHelper PUBLIC 
    ${OpenBLAS_INCLUDE_DIRS}
    ${IPOPT_LIBRARIES}
)

target_compile_definitions(IpoptHelper PRIVATE VERSION_INFO=0.1.0)
