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
include src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/depend.make

# Include the progress variables for this target.
include src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/progress.make

# Include the compile flags for this target's objects.
include src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/flags.make

src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/NNet.cpp.o: src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/flags.make
src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/NNet.cpp.o: ../src/bin/eigen/project/NNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/NNet.cpp.o"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project && /usr/bin/g++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Cpp_Eigen.dir/NNet.cpp.o -c /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/project/NNet.cpp

src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/NNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Cpp_Eigen.dir/NNet.cpp.i"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/project/NNet.cpp > CMakeFiles/Cpp_Eigen.dir/NNet.cpp.i

src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/NNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Cpp_Eigen.dir/NNet.cpp.s"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project && /usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/project/NNet.cpp -o CMakeFiles/Cpp_Eigen.dir/NNet.cpp.s

# Object files for target Cpp_Eigen
Cpp_Eigen_OBJECTS = \
"CMakeFiles/Cpp_Eigen.dir/NNet.cpp.o"

# External object files for target Cpp_Eigen
Cpp_Eigen_EXTERNAL_OBJECTS =

../target/debug/bin/Cpp_Eigen: src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/NNet.cpp.o
../target/debug/bin/Cpp_Eigen: src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/build.make
../target/debug/bin/Cpp_Eigen: src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../../target/debug/bin/Cpp_Eigen"
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Cpp_Eigen.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/build: ../target/debug/bin/Cpp_Eigen

.PHONY : src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/build

src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/clean:
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project && $(CMAKE_COMMAND) -P CMakeFiles/Cpp_Eigen.dir/cmake_clean.cmake
.PHONY : src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/clean

src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/depend:
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daisy/W/W.priv/NonBlockingProtocol /home/daisy/W/W.priv/NonBlockingProtocol/src/bin/eigen/project /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/bin/eigen/project/CMakeFiles/Cpp_Eigen.dir/depend

