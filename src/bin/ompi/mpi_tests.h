//
// Created by egrzrbr on 2019-04-28.
//

#ifndef NONBLOCKINGPROTOCOL_MPI_TESTS_H
#define NONBLOCKINGPROTOCOL_MPI_TESTS_H

#include <string>
#include <iostream>
#include <stdio.h>

#ifdef WITH_MPI

#include <mpi.h>

#endif

#ifdef WITH_BOOST_MPI

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/string.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/mpi/collectives/gather.hpp>

namespace mpi = boost::mpi;
namespace mt  = mpi::threading;


#endif

#include <memory>
#include <iostream>
#include <string>
#include <cstdio>

int randi_range(int min, int max) //range : [min, max)
{

	srand(time(NULL)); //seeding for the first time only!
	return min + rand() % ((max + 1) - min);

}

template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {

	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}


enum msg_e {
	BEFORE_DATA_E,
	COMPUTING_E,
	AFTER_DATA_E,
	SENDING_E,
	RECEIVING_E,

};

template<typename T, size_t N>
class MPI_Node {

	enum dir_e {
		IN, OUT
	};
	enum item_e {
		DST, FLG, CHK
	};

	float sleepTime;

	mpi::environment &env;
	mpi::communicator &world;

	int msg[2][3];

	T arrOut[N] = {};
	T arrIn[4][N] = {};

	int rank;

  public:
	MPI_Node(mpi::environment &env, mpi::communicator &world) : env(env), world(world) {

		rank = world.rank();

		sleepTime = randi_range(1, 10) * 1. / float(rank + 1);

		msg[OUT][DST] = rank;

		if (0) {
			std::cout << "### COMPUTE start, PROC : " << rank << std::endl;
			out_msg_compute();
			in_msg_compute();
			std::cout << "### COMPUTE end, PROC : " << rank << std::endl;


			std::cout << "### BARRIER START : COMPUTE | MSG BEFORE DATA, PROC : " << rank << std::endl;
			world.barrier();
			std::cout << "### BARRIER END : COMPUTE | MSG BEFORE DATA, PROC : " << rank << std::endl;


			std::cout << "### MSG BEFORE DATA start, PROC : " << rank << std::endl;
			out_msg_before_data();
			in_msg_before_data();
			std::cout << "### MSG BEFORE DATA end, PROC : " << rank << std::endl;

			std::cout << "### BARRIER START : MSG BEFORE DATA | BCAST DATA, PROC : " << rank << std::endl;
			world.barrier();
			std::cout << "### BARRIER END : MSG BEFORE DATA | BCAST DATA, PROC : " << rank << std::endl;
		}

		std::cout << "### DATA start, PROC : " << rank << std::endl;
		out_data();
		std::cout << "### BARRIER START : BCAST DATA OUT | BCAST DATA IN, PROC : " << rank << std::endl;
		world.barrier();
		std::cout << "### BARRIER END : BCAST DATA OUT | BCAST DATA IN, PROC : " << rank << std::endl;
		in_data();
		std::cout << "### DATA end, PROC : " << rank << std::endl;

		std::cout << "### BARRIER START : BCAST DATA IN | MSG AFTER DATA, PROC : " << rank << std::endl;
		world.barrier();
		std::cout << "### BARRIER END : BCAST DATA IN | MSG AFTER DATA, PROC : " << rank << std::endl;


		std::cout << "### MSG AFTER DATA start, PROC : " << rank << std::endl;
		out_msg_after_data();
		in_msg_after_data();
		std::cout << "### MSG AFTER DATA end, PROC : " << rank << std::endl;


	}

	void msg_info(int n, dir_e dirE, msg_e msgE) {

		if (dirE == IN)
			std::cout << ">>> ";
		else if (dirE == OUT)
			std::cout << "<<< ";

		std::cout << n << " [" << time(NULL) << "]" << " Process #" << rank;

		if (dirE == IN)
			std::cout << " is receiving ";
		else if (dirE == OUT)
			std::cout << " is sending ";

		if (msgE < SENDING_E) {
			std::cout << " msg [rank, ";

			if (msgE == BEFORE_DATA_E)
				std::cout << "BEFORE_DATA_E";
			else if (msgE == AFTER_DATA_E)
				std::cout << "AFTER_DATA_E";
			else if (msgE == COMPUTING_E)
				std::cout << "COMPUTING_E";

			std::cout << " check] " << std::endl;

		} else {

			if (msgE == SENDING_E)
				std::cout << "SENDING_E";
			else if (msgE == RECEIVING_E)
				std::cout << "RECEIVING_E";

			std::cout << " broadcast of DATA" << std::endl;
		}

	}

	bool out_msg_compute() {

		msg_info(1, OUT, COMPUTING_E);
		msg[OUT][FLG] = COMPUTING_E;
		broadcast(world, msg[OUT], 3, rank);
		sleep(sleepTime);
		for (int i = 0; i < N; i++) arrOut[i] = rank;

	}

	bool in_msg_compute() {

		msg_info(1, IN, COMPUTING_E);

		for (int proc = 0; proc < world.size(); proc++) {
			if (proc == rank) continue;

			broadcast(world, msg[IN], 3, proc);

			std::cout << "... <<< 1 [" << time(NULL) << "]"
			          << " Process #" << world.rank()
			          << " received msg[rank, COMPUTING_E, check] : {"
			          << msg[IN][DST] << "(" << proc << ")" << "," << msg[IN][FLG] << "," << msg[IN][CHK] << "}" << std::endl;

		}
	}

	void out_msg_before_data() {

		msg_info(2, OUT, BEFORE_DATA_E);

		msg[OUT][FLG] = BEFORE_DATA_E;
		msg[OUT][FLG] = N;
		broadcast(world, msg[OUT], 3, rank);

	}

	void in_msg_before_data() {

		msg_info(2, IN, BEFORE_DATA_E);

		for (int proc = 0; proc < world.size(); proc++) {
			if (proc == rank) continue;

			broadcast(world, msg[IN], 3, proc);

			std::cout << "... <<< 2 [" << time(NULL) << "]"
			          << " Process #" << world.rank()
			          << " received msg[rank, BEFORE_DATA, check] : {"
			          << msg[IN][DST] << "(" << proc << ")" << "," << msg[IN][FLG] << "," << msg[IN][CHK] << "}" << std::endl;

		}
	}

	void out_data() {

		msg_info(3, OUT, SENDING_E);

		broadcast(world, &arrOut[0], N, rank);
		

	}

	void in_data() {

		msg_info(3, OUT, RECEIVING_E);

		for (int proc = 0; proc < world.size(); proc++) {

			if (proc == rank) continue;

			broadcast(world, &arrIn[proc][0], N, proc);
			float sum = 0;
			for (int i = 0; i < N; i++) {
				sum += arrIn[proc][i];
			}

			std::cout << "... <<< 3 [" << time(NULL) << "]"
			          << " Process #" << world.rank()
			          << " received data<T, N> : sum = " << sum << ", checksum = " << msg[IN][CHK] << std::endl;

		}

	}

	void out_msg_after_data() {

		msg_info(3, OUT, AFTER_DATA_E);
		msg[OUT][FLG] = AFTER_DATA_E;
		msg[OUT][CHK] = N;
		broadcast(world, msg[OUT], 3, rank);
	}

	void in_msg_after_data() {

		msg_info(3, OUT, AFTER_DATA_E);

		for (int proc = 0; proc < world.size(); proc++) {
			if (proc == rank) continue;

			broadcast(world, msg[IN], 3, proc);

			std::cout << "... <<< 4 [" << time(NULL) << "]"
			          << " Process #" << world.rank()
			          << " received msg[rank, AFTER_DATA_E, check] : {"
			          << msg[IN][DST] << "(" << proc << ")" << "," << msg[IN][FLG] << "," << msg[IN][CHK] << "}" << std::endl;

		}
	}


};


#endif //NONBLOCKINGPROTOCOL_MPI_TESTS_H
