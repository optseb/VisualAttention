# AUTOMOC requires 2.8.6. Per-target include directories would require 2.8.8
cmake_minimum_required(VERSION 2.8.6)
PROJECT(vsDataMaker)

### Build subdirs. Also add them to include path.
#add_subdirectory(Scene3D)
#include_directories(Scene3D)

### Header file
file(GLOB SOURCE_HEADERS *.h)

### Add a per-user include directory in case that's useful
include_directories($ENV{HOME}/usr/include)
link_directories($ENV{HOME}/usr/lib)

### Executable (there is no exe for this proj; only the Brahms component)
set(TARGET worldtest CACHE TYPE STRING)
set(SOURCE ${SOURCE_HEADERS} worldtest.cpp worldframe.cpp eyeframe.cpp)
add_executable(${TARGET} ${SOURCE})
set_target_properties (${TARGET} PROPERTIES COMPILE_FLAGS "-g -DDEBUG")
target_link_libraries(${TARGET} jsoncpp)

set(TARGET_WTCO worldtest_coords CACHE TYPE STRING)
set(SOURCE_WTCO ${SOURCE_HEADERS} worldtest_coords.cpp worldframe.cpp eyeframe.cpp)
add_executable(${TARGET_WTCO} ${SOURCE_WTCO})
set_target_properties (${TARGET_WTCO} PROPERTIES COMPILE_FLAGS "-g")
target_link_libraries(${TARGET_WTCO} jsoncpp)

### Brahms Component
set(BC_TARGET vsDataMaker CACHE TYPE STRING)
set(BC_SOURCE ${SOURCE_HEADERS} vsDataMaker.cpp worldframe.cpp eyeframe.cpp)
set(SYSTEMML_INSTALL_PATH $ENV{SYSTEMML_INSTALL_PATH} CACHE PATH
    "Installation location of Brahms (e.g. ~/SystemML)")
include_directories(/usr/include/brahms)
include_directories(/var/lib/brahms/Namespace)
include_directories(${SYSTEMML_INSTALL_PATH}/BRAHMS/include)
include_directories(${SYSTEMML_INSTALL_PATH}/Namespace)
include_directories($ENV{HOME}/usr/include)
include_directories(/usr/include/eigen3)
include_directories(/usr/local/include)
link_directories(${SYSTEMML_INSTALL_PATH}/BRAHMS/bin)
add_library(${BC_TARGET} SHARED ${BC_SOURCE})
# "" rather than "lib" to give Brahms style "component.so" not "libcomponent.so":
set_target_properties(${BC_TARGET} PROPERTIES PREFIX "")
set_target_properties (${BC_TARGET} PROPERTIES COMPILE_DEFINITIONS "__GLN__")
set_target_properties (${BC_TARGET} PROPERTIES COMPILE_FLAGS "-fPIC -Werror -pthread -msse2 -O3")
##########################################################################################

### Specify libraries for Brahms Component ###############################################
target_link_libraries(${BC_TARGET} jsoncpp)
##########################################################################################
