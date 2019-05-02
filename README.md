# NonBlockingProtocol

This project is made on (fast and simple) standard cmake setup from:
- robgrzel/Examples/MKL-DNN_Eigen_Boost_OpenMPI_GoogleTests_Examples

As so its organized in following way:
- cmake-build-debug
- cmake-build-releasee
- doc
- extern_libs
- modules_cmake
- resources
- src:
  - bin:
    - boost
    - boost_ompi
    - cpp
    - eigen
    - mkl-dnn
    - ompi 
    - CMakeLists.txt
  - lib:
    - boost_python
    - CMakeLists.txt
  - CMakeLists.txt
- target:
  - debug:
    - bin
  - release:
    - bin
- test
- tmp
- CMakeLists.txt

Currently focus is set on studing non-blocking MPI capabilities, relevant examples are in:
- src/bin/ompi/mpi_tests.cpp.

Non blocking examples are:
- p2p : mpi_non_blocking_p2p.hxx
- bcast: mpi_non_blocing_bcast.hxx

