
if (DO_BOOST)
    project(Cpp_Multi_Perceptron_Boost LANGUAGES C CXX)
else ()
    project(Cpp_Multi_Perceptron LANGUAGES C CXX)
endif ()

cmake_minimum_required(VERSION 3.13)

#set ( CMAKE_BUILD_TYPE RELEASE )

message("Build EXE: ${PROJECT_NAME}")

set(BOOST_LIBS
        )

set(SOURCES
        main.std.stl.NN.layers.vararg.cpp
        NNet.hpp NNet.hxx
        DGen.hxx DGen.hpp
        MUtil.h ../MUtil.h
        CEng.hxx CEng.hpp tests.hpp
        )

if (DO_BOOST_PYTHON)
    add_executable_boost_python27(${PROJECT_NAME} "${SOURCES}" "${BOOST_LIBS}")
elseif (DO_BOOST)
    add_executable_boost(${PROJECT_NAME} "${SOURCES}" "${BOOST_LIBS}")

else ()
    add_executable(${PROJECT_NAME} ${SOURCES})
    config_bin_output(${PROJECT_NAME})
endif ()

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS ON)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_LIBRARY_PATH}/Eigen/include/eigen3)


if (DO_PYTHON)
    link_python(${PROJECT_NAME})
endif ()

link_resources(${PROJECT_NAME})

#project(cpp_example_1 languages c cxx)
#add_executable(${project_name} ${sources})
#set_target_properties(${project_name} properties cxx_standard 17 cxx_standard_required yes cxx_extensions on)


