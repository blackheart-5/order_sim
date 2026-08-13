cmake_minimum_required(VERSION 3.15)
project(lobsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Benchmarking a debug build measures the compiler, not the code.
if(MSVC)
    add_compile_options(/W4 /permissive-)
    add_compile_options($<$<CONFIG:Release>:/O2>)
else()
    add_compile_options(-Wall -Wextra)
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

# -march=native lets the compiler use this machine's instruction set. It makes
# the binary non-portable, so it is opt-in: leave it off if you plan to copy
# the build anywhere.
option(LOB_NATIVE "Optimise for the building machine's CPU" OFF)
if(LOB_NATIVE AND NOT MSVC)
    add_compile_options(-march=native)
endif()

# ---------------------------------------------------------------- core -----

add_library(lob_core STATIC src/order_book.cpp)
target_include_directories(lob_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------- tests & bench --

add_executable(lob_tests tests/test_order_book.cpp)
target_link_libraries(lob_tests PRIVATE lob_core)

add_executable(bench_latency bench/bench_latency.cpp)
target_link_libraries(bench_latency PRIVATE lob_core)

add_executable(bench_idlookup bench/bench_idlookup.cpp)
target_link_libraries(bench_idlookup PRIVATE lob_core)

enable_testing()
add_test(NAME order_book COMMAND lob_tests)

# ------------------------------------------------------- python bindings ---

# Optional: the C++ side builds and tests fine without Python present.
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND AND Python3_Interpreter_FOUND)
    # pybind11 installed via pip ships its own CMake config; ask Python where.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(PYBIND11_CMAKE_DIR)
        find_package(pybind11 CONFIG QUIET PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
    endif()
endif()

if(pybind11_FOUND)
    pybind11_add_module(lobsim python/bindings.cpp src/order_book.cpp)
    target_include_directories(lobsim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # NumPy headers are needed for the array-returning bindings.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NUMPY_INCLUDE)
        target_include_directories(lobsim PRIVATE "${NUMPY_INCLUDE}")
    else()
        message(WARNING "NumPy headers not found; the lobsim module may fail to build. "
                        "Install with: pip install numpy")
    endif()

    # Drop the Python scripts next to the module so they can be run from the
    # build directory without setting PYTHONPATH.
    foreach(script market_sim.py impact_study.py)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/${script}
                       ${CMAKE_CURRENT_BINARY_DIR}/${script} COPYONLY)
    endforeach()

    message(STATUS "Python bindings: enabled")
else()
    message(STATUS "Python bindings: disabled (pybind11 not found -- "
                   "pip install pybind11 numpy to enable)")
endif()
