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


# Produce verbose output by default.
VERBOSE = 1

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
CMAKE_SOURCE_DIR = /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin

# Include any dependencies generated for this target.
include CMakeFiles/boo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/boo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/boo.dir/flags.make

CMakeFiles/boo.dir/main.c.o: CMakeFiles/boo.dir/flags.make
CMakeFiles/boo.dir/main.c.o: /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src/main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/boo.dir/main.c.o"
	/usr/bin/gcc-8 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/boo.dir/main.c.o   -c /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src/main.c

CMakeFiles/boo.dir/main.c.i: cmake_force
	@echo "Preprocessing C source to CMakeFiles/boo.dir/main.c.i"
	/usr/bin/gcc-8 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src/main.c > CMakeFiles/boo.dir/main.c.i

CMakeFiles/boo.dir/main.c.s: cmake_force
	@echo "Compiling C source to assembly CMakeFiles/boo.dir/main.c.s"
	/usr/bin/gcc-8 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src/main.c -o CMakeFiles/boo.dir/main.c.s

# Object files for target boo
boo_OBJECTS = \
"CMakeFiles/boo.dir/main.c.o"

# External object files for target boo
boo_EXTERNAL_OBJECTS =

boo: CMakeFiles/boo.dir/main.c.o
boo: CMakeFiles/boo.dir/build.make
boo: libfoo.a
boo: CMakeFiles/boo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable boo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/boo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/boo.dir/build: boo

.PHONY : CMakeFiles/boo.dir/build

CMakeFiles/boo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/boo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/boo.dir/clean

CMakeFiles/boo.dir/depend:
	cd /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/src /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin /home/daisy/W/W.priv/NonBlockingProtocol/cmake-build-debug/src/bin/eigen/examples/CMakeFiles/_CMakeLTOTest-C/bin/CMakeFiles/boo.dir/DependInfo.cmake
.PHONY : CMakeFiles/boo.dir/depend

