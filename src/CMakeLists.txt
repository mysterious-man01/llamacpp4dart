cmake_minimum_required(VERSION 3.10)
project(llamalib VERSION 0.0.1 LANGUAGES C)

set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_policy(VERSION 3.14...3.25)

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)
set(VK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan)
set(SPIRV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spirv)
set(VK_VERSION "vulkan-sdk-1.3.275")

# Clone llama.cpp if nescessary
if(NOT EXISTS ${LLAMA_DIR})
  message(STATUS "Downloading llama.cpp...")
  execute_process(COMMAND git clone --recursive https://github.com/ggml-org/llama.cpp ${LLAMA_DIR})
endif()

# Download headers Vulkan if nescessary
if(NOT EXISTS ${VK_DIR}/Vulkan-Headers/include)
  message(STATUS "Downloading Vulkan ${VK_VERSION} SDK headers...")
  file(MAKE_DIRECTORY "${VK_DIR}")
  execute_process(COMMAND git clone --depth 1 --branch ${VK_VERSION} https://github.com/KhronosGroup/Vulkan-Headers.git ${VK_DIR}/Vulkan-Headers)
endif()
set(Vulkan_INCLUDE_DIRS "${VK_DIR}/Vulkan-Headers/include" CACHE STRING "" FORCE)
include_directories(BEFORE ${Vulkan_INCLUDE_DIRS})

# Download SPIRV headers if nescessary
if(NOT EXISTS ${SPIRV_DIR}/SPIRV-Headers/include)
  message(STATUS "Downloading SPIRV-Headers...")
  execute_process(COMMAND git clone --depth 1 --branch ${VK_VERSION} https://github.com/KhronosGroup/SPIRV-Headers.git ${SPIRV_DIR}/SPIRV-Headers)
endif()
set(SPIRV_Headers_DIR "${SPIRV_DIR}/SPIRV-Headers/include" CACHE STRING "" FORCE)
include_directories(BEFORE ${SPIRV_Headers_DIR})

# Set Vulkan on
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

# GGML options
if(ANDROID)
  find_library(Vulkan_LIBRARY vulkan REQUIRED)

  add_definitions(-DGGML_VULKAN=ON)

  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
  
  set(GGML_BLAS OFF CACHE BOOL "" FORCE)
  set(GGML_CPU ON CACHE BOOL "" FORCE)
  set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
endif()

# Build it as dynamic library
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)

# Add llama.cpp
add_subdirectory(${LLAMA_DIR})

# Define flag for dart
target_compile_definitions(llama PUBLIC DART_SHARED_LIB)

# Link Vulkan if disponible
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND ANDROID)
  if(TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan BEFORE PUBLIC ${Vulkan_INCLUDE_DIRS})
  endif()
  if(TARGET llama)
    target_link_libraries(llama PRIVATE ${Vulkan_LIBRARY})
    # for pagination compatbility on modern Android
    target_link_options(llama PRIVATE "-Wl,-z,max-page-size=16384")
  endif()
endif()