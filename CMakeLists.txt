cmake_minimum_required(VERSION 3.1)
project(libmcraw)

set(CMAKE_C_STANDARD 99)

add_executable(metadata samples/metadata.c lib/mcraw.c lib/RawData.cpp)
add_executable(export_audio samples/export_audio.c lib/mcraw.c lib/RawData.cpp)

target_link_libraries(metadata uuid)
target_link_libraries(export_audio uuid)

add_compile_options(-Wall -Wextra -pedantic -Werror -O2)
