cmake_minimum_required(VERSION 3.20)
project(tokchat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_LLAMA "Build with llama.cpp backend" ON)

add_executable(tokchat
  src/main.cpp
  src/sampling.cpp
  src/chat_format.cpp
  src/llama_backend.cpp
)

target_include_directories(tokchat PRIVATE include)

if (USE_LLAMA)
  # Expect llama.cpp checked out at third_party/llama.cpp
  add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
  target_link_libraries(tokchat PRIVATE llama)
  target_compile_definitions(tokchat PRIVATE USE_LLAMA)
endif()