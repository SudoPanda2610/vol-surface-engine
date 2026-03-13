cmake_minimum_required(VERSION 3.15)
project(cn_pricer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(cn_pricer crank_nicolson.cpp)
