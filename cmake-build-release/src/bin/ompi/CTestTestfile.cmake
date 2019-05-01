# CMake generated Testfile for 
# Source directory: /home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi
# Build directory: /home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/cmake-build-release/src/bin/ompi
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(OpenMPI_NonBlocking_test_001 "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/OpenMPI/bin/mpiexec" "-n" "1" "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/target/release/bin/OpenMPI_NonBlocking")
set_tests_properties(OpenMPI_NonBlocking_test_001 PROPERTIES  _BACKTRACE_TRIPLES "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi/CMakeLists.txt;35;add_test;/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi/CMakeLists.txt;0;")
add_test(OpenMPI_NonBlocking_test_002 "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/OpenMPI/bin/mpiexec" "-n" "4" "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/target/release/bin/OpenMPI_NonBlocking")
set_tests_properties(OpenMPI_NonBlocking_test_002 PROPERTIES  _BACKTRACE_TRIPLES "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi/CMakeLists.txt;36;add_test;/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi/CMakeLists.txt;0;")
add_test(OpenMPI_NonBlocking_test_003 "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/OpenMPI/bin/mpiexec" "-n" "8" "--oversubscribe" "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/target/release/bin/OpenMPI_NonBlocking")
set_tests_properties(OpenMPI_NonBlocking_test_003 PROPERTIES  _BACKTRACE_TRIPLES "/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi/CMakeLists.txt;37;add_test;/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/src/bin/ompi/CMakeLists.txt;0;")
subdirs("CppUtils")
