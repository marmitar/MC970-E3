cmake_minimum_required(VERSION 3.19)
project(smoothing LANGUAGES C CUDA)

# Set language standards to C17 and C++17
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# If build type is unspecified, default to Release
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "Build Configuration" FORCE)
endif()

# Enable Link Time Optimizations
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    CACHE BOOL "Enable LTO in Release")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
    CACHE BOOL "Enable LTO in RelWithDebInfo")

# Set compiler warnings
add_compile_options(
    -Wall -Wextra
    # Enable basic inlining reports in relese
    $<$<CONFIG:Release,RelWithDebInfo>:-Winline>
    # NVCC emits unresolvable warnings with -Wpedantic
    $<IF:$<COMPILE_LANGUAGE:CUDA>,-Wno-pedantic,-Wpedantic>
    # Some string and formatting footguns
    -Wformat=2 -Wformat-truncation=2 -Wformat-overflow=2 -Wstringop-overflow=4
    # Warn on implicit casts
    -Wconversion -Wsign-conversion -Wformat-signedness
    # CUDA specific warnings
    $<$<COMPILE_LANGUAGE:CUDA>:-Wreorder>
    # Enable calling __host__ function in __device__ code for constexpr contexts
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
add_compile_definitions($<$<CONFIG:Release>:_FORTIFY_SOURCE=1>)
add_compile_definitions($<$<CONFIG:RelWithDebInfo>:_FORTIFY_SOURCE=3>)

# Set the default NVIDIA architecture to 35 (oldest in CENAPAD, CUDA 11.5.0)
if (NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 35)
    # sm_35 is deprecated even in CUDA 11.5, but we can ignore these warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
endif()

## BINARIES ##

# Set number of threads for the solution
set(THREADS_PER_BLOCK "256" CACHE STRING "Number of threads for each CUDA block")
add_compile_definitions(THREADS_PER_BLOCK=${THREADS_PER_BLOCK})

# Set mask width for image smoothing
set(MASK_WIDTH "15" CACHE STRING "Width of the mask used for image smoothing")
add_compile_definitions(MASK_WIDTH=${MASK_WIDTH})

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(serial   src/smooth-serial.c)
add_executable(parallel src/smooth-parallel.cu)

target_link_libraries(serial   PRIVATE OpenMP::OpenMP_C m)
target_link_libraries(parallel PRIVATE OpenMP::OpenMP_C m)
