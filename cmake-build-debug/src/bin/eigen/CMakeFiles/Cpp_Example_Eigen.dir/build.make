# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/daisy/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/191.6707.69/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/daisy/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/191.6707.69/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daisy/W/W.priv/NonBlockingProtocol

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug

# Include any dependencies generated for this target.
include src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/depend.make

# Include the progress variables for this target.
include src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/progress.make

# Include the compile flags for this target's objects.
include src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/flags.make

src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.o: src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/flags.make
src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.o: ../src/bin/eigen/matmul_eigen.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.o"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen && /usr/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.o -c /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/matmul_eigen.cpp

src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.i"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/matmul_eigen.cpp > CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.i

src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.s"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/matmul_eigen.cpp -o CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.s

# Object files for target Cpp_Example_Eigen
Cpp_Example_Eigen_OBJECTS = \
"CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.o"

# External object files for target Cpp_Example_Eigen
Cpp_Example_Eigen_EXTERNAL_OBJECTS =

../target/debug/bin/Cpp_Example_Eigen: src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/matmul_eigen.cpp.o
../target/debug/bin/Cpp_Example_Eigen: src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/build.make
../target/debug/bin/Cpp_Example_Eigen: src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../target/debug/bin/Cpp_Example_Eigen"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Cpp_Example_Eigen.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/build: ../target/debug/bin/Cpp_Example_Eigen

.PHONY : src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/build

src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/clean:
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen && $(CMAKE_COMMAND) -P CMakeFiles/Cpp_Example_Eigen.dir/cmake_clean.cmake
.PHONY : src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/clean

src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/depend:
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daisy/W/W.priv/NonBlockingProtocol /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/bin/eigen/CMakeFiles/Cpp_Example_Eigen.dir/depend

