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

#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <Eigen/Dense>

#define RESOURCES_DATA_PATH (std::string) "resources/data"

#ifdef WITH_BOOST_PYTHON

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

	#ifdef WITH_BOOST_PYTHON
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
	constexpr size_t mB = 1000;

	{
		TIME_START

		//auto xtrainBegin = mnist->begin<DGen::data_e::TRAIN_IMG>();
		//auto xtrainEnd = mnist->end<DGen::data_e::TRAIN_IMG>();

		//auto f = std::accumulate(xtrainBegin, xtrainEnd, 0.f, [&](MNIST_IN_t in, MNIST_IN_t &x) {return in + x;});

		//std::cout << "sum = " << f / double{256} << std::endl;

		TIME_CHECK
	}

	std::srand(0);

	auto layersMaker = NNet::LayersMaker<IN_T, HL_T, B, mB, INPUT_LAYER, 20, OUTPUT_LAYER>();

	auto layersPtr = layersMaker.alloc<DEBUG1>();

	auto datagen = DGen::Datagen<HL_T, DGen::RANDOM_NORMAL>();

	//auto engine = CEng::Engine();


	auto network = NNet::Network(&layersMaker, layersPtr, &datain, &datagen);

	network.print();

	network.init<DEBUG1>();

	network.compute<DEBUG2>();

	constexpr size_t N_TRIALS = (int) 1e4;

	size_t n_trials = N_TRIALS;

	TIME_START
	while (n_trials--) {

		if (n_trials % 1000 == 0) std::cout << "TRIAL : " << n_trials << "/" << N_TRIALS << std::endl;
		network.compute<DEBUG0>();
	}
	TIME_CHECK



/*	auto start = clock();

	size_t o = 0;
	for (size_t i = 0; i < 1000; i++) {
		o += 1;
		compute<DEBUG0, Layers_Ptr_t, I, shapeNN...>(1, layersWrap, enginePtr);
	}

	std::cout << "Took " << clock() - start << " ms" << std::endl;

	std::cout << o << std::endl;
*/

	//auto * lh = (LayerHidden<int, 100, 100, mB> * ) layersWrap.layersTraining[1];
	//lh->V;
	//layersWrap->print();

	std::cout << "Done." << std::endl;

	return 0;
}
