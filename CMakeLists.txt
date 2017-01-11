cmake_minimum_required(VERSION 2.8)

# Project Name
project(mybm)

#########################################################
# RESOURCES
#########################################################
SET (RES_LST left.png right.png)

foreach(filename ${RES_LST})
	configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)    
endforeach(filename)

#########################################################
# FIND OPENCV
#########################################################
# Under Windows the system variable "OPENCV_ROOT" must be set to the location of the root directory of OpenCV.
find_package(OpenCV 2.4 REQUIRED)

#########################################################
# SOURCES
#########################################################

aux_source_directory(./ SOURCES)
add_executable(${PROJECT_NAME} ${SOURCES})

########################################################
# Linking & stuff
#########################################################

#include_directories()
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
