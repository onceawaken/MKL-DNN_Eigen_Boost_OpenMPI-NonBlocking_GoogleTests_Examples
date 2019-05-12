#include <vector>
#include <iostream>
#include <map>
#include <cstring>
#include <list>
#include <memory>
#include <chrono>
#include <thread>
#include <cassert>
#include <csignal>
#include <cmath>
#include <variant>
#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <fstream>
#include <time.h>
#include <numeric>

#include <Eigen/Dense>

//#include <pstl/execution>
//#include <pstl/algorithm>
//#include <pstl/numeric>
//#include <pstl/memory>

#define RESOURCES_DATA_PATH (std::string) "resources/data"

#ifdef WITH_BOOST

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

#endif

#include "MUtil.h"
#include "DGen.hpp"
#include "NNet.hpp"
#include "CEng.hpp"
#include "tests.hpp"

int main() {

#ifdef WITH_BOOST
	Py_Initialize();

	boost::python::numpy::initialize();
#endif


	//dgen_tests();

	std::cout << "LOAD MNIST" << std::endl;

#define IN_T float
#define HL_T float

	auto datain = DGen::LoadMnist<DEBUG0>("data");

	constexpr size_t INPUT_LAYER = datain.get_pattern_size();
	constexpr size_t OUTPUT_LAYER = datain.get_labels_size();
	constexpr size_t B = datain.get_train_batch_size();
	constexpr size_t mB = 100;
	constexpr HL_T eta = 0.3;

	{
		TIME_START

		//auto xtrainBegin = mnist->begin<DGen::data_e::TRAIN_IMG>();
		//auto xtrainEnd = mnist->end<DGen::data_e::TRAIN_IMG>();

		//auto f = std::accumulate(xtrainBegin, xtrainEnd, 0.f, [&](MNIST_IN_t in, MNIST_IN_t &x) {return in + x;});

		//std::cout << "sum = " << f / double{256} << std::endl;

		TIME_CHECK
	}

	std::srand(0);

	constexpr size_t epochs = 2;

	auto layersMaker = NNet::LayersMaker<IN_T, HL_T, B, mB, INPUT_LAYER, 100, 30, OUTPUT_LAYER>();

	auto layersPtr = layersMaker.alloc<DEBUG1>();

	layersPtr->print();

	auto datagen = DGen::Datagen<HL_T, DGen::RANDOM_NORMAL>();

	auto engine = CEng::Engine<HL_T, CEng::SEQUENTIAL_E>(eta);

	auto network = NNet::Network(layersMaker, layersPtr, engine, datain, datagen);

	network.print();

	network.init<DEBUG1>();

	network.compute<DEBUG2>(epochs);

	std::cout << "Done." << std::endl;

	return 0;
}
