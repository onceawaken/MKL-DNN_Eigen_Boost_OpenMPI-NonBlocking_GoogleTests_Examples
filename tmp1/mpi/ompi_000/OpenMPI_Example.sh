#!/bin/bash -e

set -e


OPAL_PREFIX=$PWD/extern/OpenMPI LD_LIBRARY_PATH=extern/OpenMPI/lib extern/OpenMPI/bin/mpirun -n 4 target/debug/bin/OpenMPI_NonBlocking
OPAL_PREFIX=$PWD/extern/OpenMPI LD_LIBRARY_PATH=extern/OpenMPI/lib extern/OpenMPI/bin/mpirun --oversubscribe -n 8 target/debug/bin/OpenMPI_NonBlocking

exit -1

set +e