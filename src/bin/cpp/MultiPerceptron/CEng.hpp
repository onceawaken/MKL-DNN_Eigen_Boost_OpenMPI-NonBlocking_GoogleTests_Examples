//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_CENG_HPP
#define CPP_EXAMPLE_CENG_HPP

namespace CEng {

	class Engine {

	  public:

		template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
		class InnerEngine;

		template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
		auto get_engine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, Network_Ptr_t networkPtr);

	};

	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
	class Engine::InnerEngine {

		using Prev_Linked_t = std::remove_pointer_t<Prev_Linked_Ptr_t>;
		using Next_Linked_t = std::remove_pointer_t<Next_Linked_Ptr_t>;
		static constexpr NNet::layers_e PREV_EID = Prev_Linked_t::EID;
		static constexpr NNet::layers_e NEXT_EID = Next_Linked_t::EID;
		static constexpr size_t N = Next_Linked_t::N;
		static constexpr size_t M = Next_Linked_t::M;
		static constexpr size_t B = Prev_Linked_t::B;
		static constexpr size_t mB = Prev_Linked_t::mB;

		using T_prev = typename Prev_Linked_t::T;
		using T_next = typename Next_Linked_t::T;

		Prev_Linked_Ptr_t prevLinkedPtr;
		Next_Linked_Ptr_t nextLinkedPtr;
		Network_Ptr_t networkPtr;

	  public:
		InnerEngine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, Network_Ptr_t networkPtr);

		void compute_forward(size_t epoch, size_t iter);

		void compute_update(size_t epoch, size_t iter);

		void compute_backward(size_t epoch, size_t iter);

	};

	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
	auto Engine::get_engine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, Network_Ptr_t networkPtr) {
		return InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t, Network_Ptr_t>(prevLinkedPtr, nextLinkedPtr, networkPtr);

	}

}

#include "CEng.hxx"

#endif //CPP_EXAMPLE_CENG_HPP
