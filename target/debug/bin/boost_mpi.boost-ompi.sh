#!/bin/bash -e

echo "Boost_OpenMPI_Example RUNNER..."

OPAL_PREFIX=/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/OpenMPI
LD_LIBRARY_PATH=:/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/OpenMPI/lib:/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/Boost/lib
MPI_RUN=/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/extern_libs/OpenMPI/bin/mpirun
TARGET=/home/egrzrbr/W/W.priv/HPC/NonBlockingProtocol/target/debug/bin/Boost_OpenMPI_Example

CMD="OPAL_PREFIX=$OPAL_PREFIX LD_LIBRARY_PATH=$LD_LIBRARY_PATH $MPI_RUN --oversubscribe -np  $TARGET"

echo $CMD

eval $CMD


exit -1
