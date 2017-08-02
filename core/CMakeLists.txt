#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 2.8)
#Name your project here
project(dsac)

#Sends the -std=c99 flag to the gcc compiler
add_definitions(-std=c++11 -Ofast)

# PNG++
include(FindPNG)
include_directories(${PNG_INCLUDE_DIR})

# LUA and Torch
include_directories(/usr/include/lua5.3)
include_directories(~/lib/torch/install/include/)
link_directories(~/lib/torch/install/lib/)

# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
else(NOT OpenCV_FOUND)
	message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
	message(STATUS "OpenCV Link Libraries: ${OpenCV_LIBS}")
endif(NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS} )

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
   
add_executable(train_obj 
    train_obj.cpp
    util.cpp
    dataset.cpp
    Hypothesis.cpp
    properties.cpp
    read_data.cpp
    thread_rand.cpp)

add_executable(train_score
    train_score.cpp
    util.cpp
    dataset.cpp
    Hypothesis.cpp
    properties.cpp
    read_data.cpp
    thread_rand.cpp)

add_executable(test_ransac
    test_ransac.cpp
    util.cpp
    dataset.cpp
    Hypothesis.cpp
    properties.cpp
    read_data.cpp
    thread_rand.cpp)

add_executable(test_ransac_softam
    test_ransac_softam.cpp
    util.cpp
    dataset.cpp
    Hypothesis.cpp
    properties.cpp
    read_data.cpp
    thread_rand.cpp)

add_executable(train_ransac
    train_ransac.cpp
    util.cpp
    dataset.cpp
    Hypothesis.cpp
    properties.cpp
    read_data.cpp
    thread_rand.cpp)

add_executable(train_ransac_softam
    train_ransac_softam.cpp
    util.cpp
    dataset.cpp
    Hypothesis.cpp
    properties.cpp
    read_data.cpp
    thread_rand.cpp)
  
target_link_libraries(train_obj ${PNG_LIBRARY} ${OpenCV_LIBS} luajit) 
target_link_libraries(train_score ${PNG_LIBRARY} ${OpenCV_LIBS} luajit)
target_link_libraries(test_ransac ${PNG_LIBRARY} ${OpenCV_LIBS} luajit)
target_link_libraries(train_ransac ${PNG_LIBRARY} ${OpenCV_LIBS} luajit)
target_link_libraries(test_ransac_softam ${PNG_LIBRARY} ${OpenCV_LIBS} luajit)
target_link_libraries(train_ransac_softam ${PNG_LIBRARY} ${OpenCV_LIBS} luajit)
