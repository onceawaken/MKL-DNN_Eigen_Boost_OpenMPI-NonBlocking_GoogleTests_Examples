

if (OMPI_FOUND)


    add_subdirectory(CppUtils)


    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
        project(OpenMPI_NonBlocking CXX)

        set(TARGET ${PROJECT_NAME})

        ppr_blue("Build EXE: ${TARGET}")

        include_directories(CppUtils)

        set(SOURCES
                main.cpp
                main.mpi.cpp main.mpi.h
                MpiMaster.cpp MpiMaster.h
                MpiSlave.cpp MpiSlave.h
                ACalc.cpp ACalc.h
                AData.cpp AData.h
                )

        add_executable_mpi(${TARGET} "${SOURCES}")

        target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=3 OPAL_PREFIX=$ENV{OPAL_PREFIX} LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})

        # Add a test:
        enable_testing()

        get_runtime_output_dir(${TARGET})


        add_test(NAME ${PROJECT_NAME}_test_001 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${RUNTIME_OUTPUT_DIR}/${PROJECT_NAME})
        add_test(NAME ${PROJECT_NAME}_test_002 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${RUNTIME_OUTPUT_DIR}/${PROJECT_NAME})
        add_test(NAME ${PROJECT_NAME}_test_003 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 --oversubscribe ${RUNTIME_OUTPUT_DIR}/${PROJECT_NAME})
    endif ()
endif ()

