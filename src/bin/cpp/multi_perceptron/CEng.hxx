//
// Created by egrzrbr on 2019-04-06.
//


namespace CEng {

	template<debug_e DEBUG_E, activation_e ACTIVATION_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	Engine<DEBUG_E, ACTIVATION_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>
	::Engine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr) :
			prevLinkedPtr(prevLinkedPtr),
			nextLinkedPtr(nextLinkedPtr) {

		type_assert_ptr(Prev_Linked_Ptr_t);
		type_assert_ptr(Next_Linked_Ptr_t);

	}

	template<debug_e DEBUG_E, activation_e ACTIVATION_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Engine<DEBUG_E, ACTIVATION_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>
	::compute_forward() {

		const auto &X = prevLinkedPtr->get_V();
		auto &b = nextLinkedPtr->get_b();
		auto &V = nextLinkedPtr->get_V();
		auto &dV = nextLinkedPtr->get_dV();
		const auto &W = nextLinkedPtr->get_W();
		const auto &Theta = nextLinkedPtr->get_Theta();

		if constexpr (DEBUG_E) {
			std::cout << "[FORWARD] : ... N=" << N << " M=" << M << std::endl;
			std::cout << "PREV -> ";
			prevLinkedPtr->print_name();
			std::cout << "NEXT -> ";
			nextLinkedPtr->print_name();
		}

		for (int mb = 0; mb < mB; mb++) {
			b[mb] = W * X[mb] - Theta;
		}
		for (int mb = 0; mb < mB; mb++) {
			if constexpr(ACTIVATION_E == SIGMOID_E) {
				V[mb] = 1 / (1 + Eigen::exp(-b[mb].array()));
			}
		}
		for (int mb = 0; mb < mB; mb++) {
			if constexpr(ACTIVATION_E == SIGMOID_E) {
				dV[mb] = V[mb].array() - (1 - V[mb].array());
			}
		}

		if constexpr (isOutputLayer) {
			const auto &Z = nextLinkedPtr->get_Z();
			auto &Delta = nextLinkedPtr->get_Delta();

			if constexpr (DEBUG_E) {
				std::cout << "[COMPUTE DELTAS] : ... N=" << N << " M=" << M << std::endl;
				std::cout << "OUTPUT -> ";
				nextLinkedPtr->print_name();
			}

			//Delta[HL][:M, :mB] = (V['Z'][:M, :mB] - V[HL][:M, :mB]) * dV[HL][:M, :mB]
			for (int mb = 0; mb < mB; mb++){
				Delta[mb] = (Z[mb] - V[mb]).array() * dV[mb].array();
			}



		}

	}

	template<debug_e DEBUG_E, activation_e ACTIVATION_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Engine<DEBUG_E, ACTIVATION_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>
	::compute_backward() {

		if constexpr (PREV_EID == NNet::LAYER_OUTPUT) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[BACKWARD FROM OUTPUT] : ..." << std::endl;
				prevLinkedPtr->print_name();
				nextLinkedPtr->print_name();
			}

		} else if constexpr (PREV_EID == NNet::LAYER_HIDDEN) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[BACKWARD FROM HIDDEN] : ..." << std::endl;
				prevLinkedPtr->print_name();
				nextLinkedPtr->print_name();
			}

		}

	}

}

