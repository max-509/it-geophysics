cmake_minimum_required(VERSION 3.10)
project(CoherentSumModule)

set(Python_ADDITIONAL_VERSIONS 3)

find_package(Threads REQUIRED)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG -fno-math-errno -std=c++11 -fopenmp -Wall -Wno-ignored-attributes")

include_directories(${PYTHON_INCLUDE_DIRS})

exec_program(${PYTHON_EXECUTABLE} 
	ARGS "-c \"import numpy; print(numpy.get_include())\"" 	
	OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
	RETURN_VALUE NUMPY_NOT_FOUND) 
if(NUMPY_NOT_FOUND)
	message(FATAL_ERROR "NumPy not found")
endif()

include_directories(${NUMPY_INCLUDE_DIR})

add_library(coherent_summation 
SHARED src/module.cpp)

set_target_properties(coherent_summation 
	PROPERTIES PREFIX "" 
	OUTPUT_NAME "${PROJECT_NAME}" 
	LINKER_LANGUAGE CXX)
