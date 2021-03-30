CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(n3ldg_plus CXX)

if(CMAKE_BUILD_TYPE MATCHES debug)
    ADD_DEFINITIONS(-DN3LDG_DEBUG)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -O1" )
elseif(CMAKE_BUILD_TYPE MATCHES profile)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pg -DNDEBUG" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -Ofast -march=native" )
endif()

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(include/fmt/include)
ADD_SUBDIRECTORY(include/fmt)
ADD_LIBRARY(n3ldg_plus STATIC include/n3ldg-plus/n3ldg-plus.cc)
TARGET_LINK_LIBRARIES(n3ldg_plus fmt)
