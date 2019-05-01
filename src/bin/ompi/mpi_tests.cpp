//
// Created by egrzrbr on 2019-04-28.
//

#include "mpi_tests.h"


namespace mpi = boost::mpi;

int test_boost_mpi_broadcast(mpi::environment &env, mpi::communicator &world) {

	const int rank = world.rank();
	const bool isMaster = world.rank() == 0;

	std::string value[4];

	if (isMaster) {
		value[0] = "Hello, World!";
	} else {
		value[rank] = std::to_string(rank);
	}

	for (int i = 0; i < 4; i++) {
		broadcast(world, value[i], i);

		std::cout << "Process #" << world.rank() << " says " << value[i] << std::endl;
	}

	return 0;
}

int test_boost_mpi_master_gather_value(mpi::environment &env, mpi::communicator &world) {

	std::cout << world.size() << std::endl;

	int rank = world.rank();
	int isMaster = rank == 0;

	int value = rank;

	if (isMaster) {
		std::vector<int> all_numbers;

		gather(world, value, all_numbers, 0);

		for (int proc = 0; proc < world.size(); ++proc)
			std::cout << "Process #" << proc << " thought of "
			          << all_numbers[proc] << std::endl;

	} else {
		gather(world, value, 0);
	}

	return 0;
}

template<typename T, class Iter>
auto vec_sum(Iter begin, Iter end) {

	return std::accumulate(begin, end, T{0}, [](T in, T x) { return in + x; });
	// every process sums it’s chunk
}


int test_boost_mpi_local_global_rank(mpi::environment &env, mpi::communicator &world) {


	MPI_Comm shmcomm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
	                    MPI_INFO_NULL, &shmcomm);
	int shmrank;
	MPI_Comm_rank(shmcomm, &shmrank);

	std::cout << "I am process " << world.rank() << "/" << world.size() << ", shmcomm=" << shmcomm << ", shmrank=" << shmrank << "." << std::endl;


	return 0;
}


void test_boost_mpi_broadcast_master_to_all(mpi::environment &env, mpi::communicator &world) {

	const int root = 0;
	const int rank = world.rank();
	const bool isMaster = world.rank() == root;

	constexpr size_t N = (int) 1e4;


	MPI_Node<float, N>(env, world);


}

enum tag_e {
	TAG_REQUEST_E,
	TAG_IRECV_E

};

enum msg_flag_e {

	BEGIN_E = INT_MIN,
	END_E = INT_MAX,

};

/*
 * Client-Centric Consistency Models
   - These models assume that clients connect to different replicas at each time
   - The models ensure that whenever a client connects to a replica,
     the replica is bought up to date with the replica that the client accessed previously
 */

template<typename T, size_t N, typename std::enable_if_t<(N > 1e5), T> * = nullptr>
auto make_arr() {

	return std::vector<T>(N, 0);

}

template<typename T, size_t N, typename std::enable_if_t<(N <= 1e5), T> * = nullptr>
auto make_arr() {

	return std::array<T, N>();

}

void mpi_nonblocking_ring(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);
	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	constexpr size_t N = (size_t) 1e8;

	auto value = make_arr<int, N + 4>();

	std::fill(std::begin(value) + 2, std::end(value) - 2, rank + 1);
	int checksum = std::accumulate(std::begin(value) + 2, std::end(value) - 2, int{0});

	value[0] = BEGIN_E;
	value[1] = checksum;
	value[N + 3 - 1] = checksum;
	value[N + 4 - 1] = END_E;

	int next = (rank + 1) % nproc;
	int prev = (rank + nproc - 1) % nproc;
	MPI_Request sreq;
	MPI_Isend(&value[0], value.size(), MPI_INT, next, TAG_REQUEST_E, MPI_COMM_WORLD, &sreq);
	int fprobe = 0;
	int ftest = 0;
	MPI_Message msg;
	MPI_Status stest, sprobe;
	do {
		if (!ftest) {
			MPI_Test(&sreq, &ftest, &stest);
			if (ftest) {
				printf("Isend : [ %i<%i> >>> %i ]\n", rank, TAG_REQUEST_E, next);
			} else {
				printf("Test  : [ %i<%i> --> %i ]\n", rank, TAG_REQUEST_E, next);
			}
		}
		if (!fprobe) {
			int err = MPI_Improbe(prev, TAG_REQUEST_E, MPI_COMM_WORLD, &fprobe, &msg, &sprobe);
			if (fprobe) {
				auto input = make_arr<int, N + 4>();
				MPI_Imrecv(&input[0], input.size(), MPI_INT, &msg, &sreq);
				printf("Irecv : [ %i<%i> <<< %i ] : RCV input { PROBE[SRC:%i, TAG:%i, ERR:%i], DATA[ %i | %i | %i | %i ] }\n",
				       rank, TAG_REQUEST_E, prev, sprobe.MPI_SOURCE, sprobe.MPI_TAG, sprobe.MPI_ERROR,
				       input[0], input[1], input[N + 3 - 1], input[N + 4 - 1]);
			} else {
				printf("Probe : [ %i<%i> <-- %i ]\n", rank, TAG_REQUEST_E, prev);
			}
		}
	} while (fprobe == 0 || ftest == 0);
	MPI_Finalize();
}

void test_isend_request(MPI_Request &sreq, int &ftests, MPI_Status &stest, int rank, int target) {
	/*
	 Use: MPI_Request_get_status - that will preserve request status when request is complete.
	 Reason: MPI_Test - that will clean request when its complete (flag ftest is set to true).
	*/

	int ftest = 0;
	MPI_Request_get_status(sreq, &ftest, &stest);
	if (ftest) {
		ftests += 1;
		printf(" >>>>> [ %i/%i ]       : SEND data { STATUS[SRC:%i, TAG:%i, ERR:%i] }\n",
		       rank, target, stest.MPI_SOURCE, stest.MPI_TAG, stest.MPI_ERROR);
		MPI_Test(&sreq, &ftest, &stest);
	} else {
		//printf(" ----> [ %i/%i ]       : test send { TAG[%i] } \n", rank, target, TAG_REQUEST_E);
	}

}

int probe_irecv_request(MPI_Request &sreq, int &fprobes, MPI_Status &sprobe, MPI_Message &msg, int rank, int sender) {

	int fprobe = 0;

	int err = MPI_Improbe(sender, TAG_REQUEST_E, MPI_COMM_WORLD, &fprobe, &msg, &sprobe);
	if (fprobe) {
		printf("       [ %i/%i ] <---- : probe recv { true, TAG[%i] } \n", sender, rank, TAG_REQUEST_E);

		fprobes += 1;
	} else {
		//printf("       [ %i/%i ] <---- : probe recv { false, TAG[%i] } \n", sender, rank, TAG_REQUEST_E);
	}

	return fprobe;
}

int test_ibcast() {

	int rank, size;
	int data[(int) 1e5];

	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("MPI start %d/%d\n", rank, size);

	if (rank == 0) {
		sleep(1);
		std::fill(std::begin(data), std::end(data), 1);
		printf("MPI %d/%d bcast\n", rank, size);
		MPI_Request req;
		MPI_Status status;
		MPI_Ibcast(&data, 1e5, MPI_INT, 0, MPI_COMM_WORLD, &req);

		int flag = 0;
		while (flag == 0) {
			MPI_Test(&req, &flag, &status);
		}
		std::cout << "After wait" << std::endl;

	} else {
		int flag = 0;
		MPI_Request req = MPI_REQUEST_NULL;
		MPI_Status status;
		MPI_Ibcast(&data, 1e5, MPI_INT, 0, MPI_COMM_WORLD, &req);
		while (flag == 0) {
			MPI_Request_get_status(req, &flag, &status);
			usleep(100 * 1000);
		}
		// MPI_Bcast can be done!
		//MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
		printf("MPI %d/%d recv bcast data: %d\n", rank, size, data[0]);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}


int test_ibcast_all_to_all() {

	int rank, size;
	int data[(int) 1e5];
	int recv[10][int(1e5)];

	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("MPI start %d/%d\n", rank, size);

	sleep(1);
	std::fill(std::begin(data), std::end(data), 1);
	printf("MPI %d/%d bcast\n", rank, size);
	MPI_Request reqSend, reqRecv[10];
	MPI_Status sSend;
	MPI_Status sRecv[10];
	MPI_Ibcast(&data, 1e5, MPI_INT, rank, MPI_COMM_WORLD, &reqSend);

	for (int i = 0; i < size; i++) {
		MPI_Ibcast(&recv[i][0], 1e5, MPI_INT, i, MPI_COMM_WORLD, &reqRecv[i]);
	}

	int ftest = 0, fprobes = 0, fprobe[10];

	std::fill(std::begin(fprobe), std::end(fprobe), 0);

	while (ftest == 0 || fprobes < size) {

		if (!ftest) {
			MPI_Test(&reqSend, &ftest, &sSend);
		}

		if (fprobes < size) {
			for (int i = 0; i < size; i++) {
				MPI_Request_get_status(reqRecv[i], &fprobe[i], &sRecv[i]);
				if (fprobe[i]) {
					printf("MPI %d >>> %d Recv bcast data[size:%d]: %d\n", i, rank, size, data[0]);
					fprobes++;
				}
			}
		}
	}
	// MPI_Bcast can be done!
	//MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}

template<typename T, size_t N>
int irecv_request(MPI_Request &sreq, MPI_Status &sprobe, MPI_Message &msg, int rank) {

	auto input = make_arr<T, N + 4>();
	MPI_Imrecv(&input[0], input.size(), MPI_INT, &msg, &sreq);
	printf("       [ %i/%i ] <<<<< : RECV data { STATUS[SRC:%i, TAG:%i, ERR:%i], DATA[ %i | %i | %i | %i ] }\n",
	       sprobe.MPI_SOURCE, rank, sprobe.MPI_SOURCE, sprobe.MPI_TAG, sprobe.MPI_ERROR,
	       input[0], input[1], input[N + 3 - 1], input[N + 4 - 1]);
}

template<class container_t, size_t N>
int ibcast_request(MPI_Request &sreq, MPI_Status &sprobe, MPI_Message &msg, int rank, container_t input) {


	printf("       [ %i/%i ] <<<<< : RECV data { STATUS[SRC:%i, TAG:%i, ERR:%i], DATA[ %i | %i | %i | %i ] }\n",
	       sprobe.MPI_SOURCE, rank, sprobe.MPI_SOURCE, sprobe.MPI_TAG, sprobe.MPI_ERROR,
	       input[0], input[1], input[N + 3 - 1], input[N + 4 - 1]);
}

void mpi_nonblocking_all_to_all(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	constexpr size_t N = (size_t) 1e8;

	std::cout << "size [MB] : " << sizeof(int) * N / (1024 * 1024) << std::endl;

	auto value = make_arr<int, N + 4>();

	std::fill(std::begin(value) + 2, std::end(value) - 2, rank + 1);
	int checksum = std::accumulate(std::begin(value) + 2, std::end(value) - 2, int{0});

	value[0] = BEGIN_E;
	value[1] = checksum;
	value[N + 3 - 1] = checksum;
	value[N + 4 - 1] = END_E;

	MPI_Request sreq[32]{};
	for (int proc = 0; proc < nproc; proc++) {
		MPI_Isend(&value[0], value.size(), MPI_INT, proc, TAG_REQUEST_E, MPI_COMM_WORLD, &sreq[proc]);
	}

	int fprobe = 0;
	int fprobes = 0;
	int ftests = 0;

	MPI_Message msg;
	MPI_Status stest, sprobe;
	do {
		if (ftests < nproc) {
			for (int i = 0; i < nproc; i++) {
				test_isend_request(sreq[i], ftests, stest, rank, i);
			}
		}

		if (fprobes < nproc) {
			for (int i = 0; i < nproc; i++) {
				fprobe = probe_irecv_request(sreq[i], fprobes, sprobe, msg, rank, i);
				if (fprobe) {
					irecv_request<int, N>(sreq[i], sprobe, msg, rank);
				}
			}
		}
	} while (fprobes < nproc || ftests < nproc);

	MPI_Finalize();

}

void mpi_nonblocking_broadcast_all_to_all(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	constexpr size_t N = (size_t) 1e8;

	std::cout << "size [MB] : " << sizeof(int) * N / (1024 * 1024) << std::endl;

	auto value = make_arr<int, N + 4>();

	std::fill(std::begin(value) + 2, std::end(value) - 2, rank + 1);
	int checksum = std::accumulate(std::begin(value) + 2, std::end(value) - 2, int{0});

	value[0] = BEGIN_E;
	value[1] = checksum;
	value[N + 3 - 1] = checksum;
	value[N + 4 - 1] = END_E;

	MPI_Request sreq[32]{};
	MPI_Ibcast(&value[0], value.size(), MPI_INT, rank, MPI_COMM_WORLD, &sreq[rank]);


	auto input = make_arr<int, N + 4>();
	auto inputs = std::vector<decltype(input)>(nproc, input);

	for (int i = 0; i < nproc; i++) {
		MPI_Ibcast(&inputs[i][0], inputs[i].size(), MPI_INT, i, MPI_COMM_WORLD, &sreq[i]);
	}

	int fprobe = 0;
	int fprobes = 0;
	int ftest = 0;

	MPI_Message msg;
	MPI_Status stest, sprobe;

	//goto label;
	do {
		if (!ftest) {
			test_isend_request(sreq[rank], ftest, stest, rank, rank);
			if (ftest){

			}
		}

		if (fprobes < nproc) {
			for (int i = 0; i < nproc; i++) {
				fprobe = probe_irecv_request(sreq[i], fprobes, sprobe, msg, rank, i);
				if (fprobe) {
					//ibcast_request<decltype(input), N>(sreq[i], sprobe, msg, rank, inputs[i]);
				}
			}
		}
	} while (fprobes < nproc || ftest == 0);

	label:

	MPI_Finalize();

}


int main(int argc, char *argv[]) {

	//mpi_nonblocking_all_to_all(argc, argv);

	//mpi_nonblocking_broadcast_all_to_all(argc, argv);

	//test_ibcast();

	test_ibcast_all_to_all();

	return 0;

	mpi::environment env;
	mpi::communicator world;


	//test_boost_mpi_broadcast(env, world);
	//test_boost_mpi_local_global_rank(env, world);
	test_boost_mpi_broadcast_master_to_all(env, world);

}