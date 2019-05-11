#!/bin/csh
setenv TBBROOT "/mnt/c/My/W/W.priv/Shared/src/tbb" #
setenv tbb_bin "/mnt/c/My/W/W.priv/Shared/src/tbb/build/linux_intel64_gcc_cc5.4.0_libc2.23_kernel4.4.0_debug" #
if (! $?CPATH) then #
    setenv CPATH "${TBBROOT}/include" #
else #
    setenv CPATH "${TBBROOT}/include:$CPATH" #
endif #
if (! $?LIBRARY_PATH) then #
    setenv LIBRARY_PATH "${tbb_bin}" #
else #
    setenv LIBRARY_PATH "${tbb_bin}:$LIBRARY_PATH" #
endif #
if (! $?LD_LIBRARY_PATH) then #
    setenv LD_LIBRARY_PATH "${tbb_bin}" #
else #
    setenv LD_LIBRARY_PATH "${tbb_bin}:$LD_LIBRARY_PATH" #
endif #
 #
