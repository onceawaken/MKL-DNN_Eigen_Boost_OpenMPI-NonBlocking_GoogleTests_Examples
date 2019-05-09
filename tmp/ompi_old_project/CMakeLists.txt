

if (OMPI_FOUND)

    add_subdirectory(CppUtils)

    # MPI

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

    set(MPI_NP 4)

    add_executable_mpi(${TARGET} ${MPI_NP} ${SOURCES})

    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=3 OPAL_PREFIX=$ENV{OPAL_PREFIX} LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})

    #    # Add a test:
    #    enable_testing()
    #
    #    get_runtime_output_dir(${TARGET})
    #
    #    add_test(NAME ${PROJECT_NAME}_test_001 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${RUNTIME_OUTPUT_DIR}/${PROJECT_NAME})
    #    add_test(NAME ${PROJECT_NAME}_test_002 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${RUNTIME_OUTPUT_DIR}/${PROJECT_NAME})
    #    add_test(NAME ${PROJECT_NAME}_test_003 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 --oversubscribe ${RUNTIME_OUTPUT_DIR}/${PROJECT_NAME})

    #  MPI

    project(OpenMPI_Tests CXX)

    set(TARGET ${PROJECT_NAME})

    ppr_blue("Build EXE: ${TARGET}")

    include_directories(CppUtils)

    set(SOURCES
            mpi_tests.cpp mpi_tests.h
            )

    set(MPI_NP 4)

    add_executable_mpi(${TARGET} ${MPI_NP} ${SOURCES})

    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG=4 OPAL_PREFIX=$ENV{OPAL_PREFIX} LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH})


    # Boost MPI


    project(OpenMPI_Tests_Boost CXX)

    set(TARGET ${PROJECT_NAME})

    ppr_blue("Build EXE: ${TARGET}")

    set(BOOST_LIBS
            threading
            )

    set(SOURCES
            mpi_tests.cpp mpi_tests.h
            )

    add_executable_boost_ompi(${TARGET} ${MPI_NP} "${SOURCES}" ${BOOST_LIBS})


endif ()

