cmake_minimum_required(VERSION 3.5)
project(Convolution)

include_directories(.)
include_directories(src)

add_library(conv src/convolution.c)

add_executable(${PROJECT_NAME}_example examples/conv_example.c)

add_executable(${PROJECT_NAME}_test tests/test_padding.c)

# add_subdirectory("ext")
find_library(fftw fftw3)
target_link_libraries(${PROJECT_NAME}_example conv ${fftw})
target_link_libraries(${PROJECT_NAME}_test conv ${fftw})

# for windows
# set_property(TARGET bar PROPERTY
#              IMPORTED_IMPLIB "c:/path/to/bar.lib")
