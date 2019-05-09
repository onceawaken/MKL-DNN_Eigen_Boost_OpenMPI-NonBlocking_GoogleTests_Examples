//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_NNET_HXX
#define CPP_EXAMPLE_NNET_HXX

#include "CEng.hpp"
#include "NNet.hpp"

namespace NNet {

	template<typename T>
	LayerBase::LayerBase(T l, size_t N, size_t M, const char *type) : m_shape{N, M}, m_type(type) {

		make_name(l);

		std::cout << "LayerBase : name = "
		          << m_name << std::endl;
	}

	void LayerBase::make_name(size_t l) {

		m_name = "[" + m_type + " : " + std::to_string(l) + "]:(" + std::to_string(m_shape.first) + "," +
		         std::to_string(m_shape.second) + ")";
	}

	auto LayerBase::get_name() {

		return m_name;
	}

	void LayerBase::print() {

		std::cout << m_name << std::endl;
	}

	void LayerBase::print_name() {

		print();

	}

	LayerNull::LayerNull(size_t id) : LayerBase(id, 0, 0, NAME) {

	}

	auto LayerNull::get_next() {

		return nullptr;
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ... shapeNN>
	LayerHidden<T, mB, N, M, shapeNN...>::LayerHidden(size_t id, const char *_type) : LayerBase(id, N, M, _type) {

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ... shapeNN>
	void LayerHidden<T, mB, N, M, shapeNN...>::print() {

		LayerBase::print();

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ...shapeNN>
	LayerHidden<T, mB, N, M, shapeNN...>::LayerHidden(size_t id) : LayerHidden(id, NAME) {

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ...shapeNN>
	void LayerHidden<T, mB, N, M, shapeNN...>::print_W() {

		for (int i = 0; i < M * N; i++) {
			std::cout << W[i] << std::endl;
		}

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ...shapeNN>
	void LayerHidden<T, mB, N, M, shapeNN...>::print_b() {

		for (int i = 0; i < M * mB; i++) {
			std::cout << b[i] << std::endl;
		}
	}

	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	LayerOutput<T, B, mB, N, M>::LayerOutput(size_t id) : LayerHidden<T, mB, N, M>(id, type) {

		std::cout << "... <INFO> Layer Output : classifiers [Z] : size : " << Z.size() << std::endl;

	}

	template<typename T, size_t B, size_t mB, size_t N>
	LayerInput<T, B, mB, N>::LayerInput(size_t id) : LayerBase(id, 0, N, NAME) {

		if (V.size()) {
			std::cout << "... <INFO> : Layer Input allocated [V] : size : " << V.size() << std::endl;
		} else {
			std::cout << "... <ERROR> : Layer Input not allocated [V]..." << std::endl;
		}

	}

	template<typename T, size_t B, size_t mB, size_t N>
	void LayerInput<T, B, mB, N>::print() {

		LayerBase::print();
	}

	template<class Curr_t, class Next_t>
	Node<Curr_t, Next_t>::Node(size_t l, Curr_Ptr_t curr, Next_Ptr_t next) : l(l), m_curr(curr), m_next(next) {

		assert(m_curr != nullptr);
		assert(m_next != nullptr);

	}

	template<class Curr_t, class Next_t>
	Node<Curr_t, Next_t>::Node(size_t l, Next_Ptr_t next) : l(l), m_next(next) {

		m_curr = new Curr_t(l);

		assert(m_curr != nullptr);
		assert(m_next != nullptr);

	}

	template<class Curr_t, class Next_t>
	void Node<Curr_t, Next_t>::print() {

		m_curr->print();
		m_next->print();
	}

	template<class Curr_t, class Next_t>
	auto Node<Curr_t, Next_t>::get_next() {

		return m_next->get_next();
	}

	template<class Curr_t, class Next_t>
	void Node<Curr_t, Next_t>::print_name() {

		m_curr->print();

	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	Linked<Curr_t, Next_Linked_Ptr_t>
	::Linked(Curr_Ptr_t curr, Next_Linked_Ptr_t next, size_t l) : l(l) {

		m_head = new node_t(l, curr, next);

		assert(m_head != nullptr);
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	Linked<Curr_t, Next_Linked_Ptr_t>
	::Linked(Next_Linked_Ptr_t next, size_t l) : l(l) {

		m_head = new node_t(l, next);
		assert(m_head != nullptr);

	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	void Linked<Curr_t, Next_Linked_Ptr_t>
	::print() {

		if (l == 0) std::cout << "## LINKED PRINT :..." << std::endl;

		if (m_head != nullptr) {
			m_head->print();
		}
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	void Linked<Curr_t, Next_Linked_Ptr_t>
	::print_name() {

		m_head->m_curr->print();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_name() {

		return m_head->m_curr->get_name();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_curr() {

		return m_head->m_curr;
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_next() {

		return m_head->m_next;

	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	void Linked<Curr_t, Next_Linked_Ptr_t>
	::print_V() {

		std::cout << get_name() << " : feed [V] : " << std::endl;
		for (const auto &a : m_head->m_curr->V) {
			std::cout << a << std::endl;
		}

	}

	void LayersBase::print() {

	}

	LayersBase::LayersBase() {

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::alloc(size_t l) {

		auto next = new LayerNull(l + 1);

		if constexpr(DEBUG_E) obj_assert_ptr(next);

		assert(next != nullptr);

		auto link = new Linked<LayerOutput<T, B, mB, N, M>, decltype(next)>(next, l);

		assert(link != nullptr);

		return link;

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M, size_t K, size_t ... nextShapeNN>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::alloc(size_t l) {

		auto next = alloc<DEBUG_E, M, K, nextShapeNN...>(l + 1);

		assert(next != nullptr);

		if constexpr(DEBUG_E) obj_assert_ptr(next);

		auto link = new Linked<LayerHidden<T, mB, N, M>, decltype(next)>(next, l);

		assert(link != nullptr);

		return link;

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::alloc() {

		std::cout << "#### LAYERS : ALLOC : START >>>" << std::endl;

		auto next = alloc<DEBUG_E, I, shapeNN...>(1);

		if constexpr(DEBUG_E) obj_assert_ptr(next);

		return new Linked<LayerInput<T_IN, B, mB, I>, decltype(next)>(next, 0);

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::print(const Linked_Ptr_t layersPtr) {

		type_assert_ptr(Linked_Ptr_t);

		layersPtr->print();

		return *this;
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::LayersMaker() {

		std::cout << "#### LAYERS : CONSTRUCT >>>" << std::endl;

		nMiniBatch = mB;
		nFullBatch = B;
		//init_random_bitmap<DEBUG1, T_IN, I, mB>(input->get_curr()->V);

		std::cout << m_shapeNN << std::endl;

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::Network(Layers_Ptr_t layersPtr, Linked_Ptr_t linkedPtr, Datain_Ptr_t datainPtr, Datagen_Ptr_t datagenPtr) :
			layersPtr(layersPtr), linkedPtr(linkedPtr), datainPtr(datainPtr), datagenPtr(datagenPtr) {

		type_assert_ptr(Layers_Ptr_t);
		type_assert_ptr(Datain_Ptr_t);
		type_assert_ptr(Datagen_Ptr_t);
		type_assert_ptr(Linked_Ptr_t);

		if constexpr(0) {
			print_sizeof(layersPtr)
			print_sizeof(linkedPtr)
			print_sizeof(datainPtr)
			print_sizeof(datagenPtr)
		}

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::forward(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l) {
		/* for l in range(1, L):
		 *     N, M = curr.shape
		 *           curr                              curr           prev             curr
		 *     b[l][:M, :mB] = einsum('ij,jm->im', W[l][:M,:N], V[l-1][:N, :mB]) - Theta[l][:M]
		 *     V[l][:M, :mB] = g(b[l][:M, :mB])
		 *    dV[l][:M, :mB] = dg(V[l][:M, :mB])
		 *
		 */

		if constexpr(DEBUG_E) type_assert_ptr(Next_Linked_Ptr_t);
		if constexpr(DEBUG_E) type_assert_ptr(Prev_Linked_Ptr_t);

		static auto enginePtr = CEng::Engine<DEBUG_E, CEng::SIGMOID_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>(prevLinkedPtr, nextLinkedPtr);

		enginePtr.compute_forward();

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::backward(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l) {

		if constexpr(DEBUG_E) type_assert_ptr(Next_Linked_Ptr_t);
		if constexpr(DEBUG_E) type_assert_ptr(Prev_Linked_Ptr_t);

		static auto enginePtr = CEng::Engine<DEBUG_E, CEng::SIGMOID_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>(prevLinkedPtr, nextLinkedPtr);

		enginePtr.compute_backward();

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::compute(Curr_Linked_Ptr_t curr, size_t l) {

		if constexpr(DEBUG_E) type_assert_ptr(Curr_Linked_Ptr_t);

		using Curr_Linked_t = std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr bool HAS_NEXT = Curr_Linked_t::HAS_NEXT;
		constexpr layers_e EID = Curr_Linked_t::EID;
		auto next = curr->get_next();

		if constexpr (HAS_NEXT) {

			if constexpr(EID == LAYER_INPUT) {
				if constexpr (DEBUG_E > DEBUG0) {
					std::cout << "[RESAMPLE INPUT]" << std::endl;
					curr->print_name();

				}
				/* 1. Resample mB minibatch from B images xTrain
				 * 2. Match mB minibatch from B labels zTrain
				 *
				 */
			}

			forward<DEBUG_E>(curr, next, l);
			compute<DEBUG_E>(next, l + 1);
			backward<DEBUG_E>(next, curr, l);

		}

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::compute() {

		if constexpr (DEBUG_E) {
			std::cout << "#### LAYERS : COMPUTE : START >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
		}

		compute<DEBUG_E>(linkedPtr, 0);

	}

	template<class Layers_Ptr_t, class Linked_Ptr, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr, Datain_Ptr_t, Datagen_Ptr_t>
	::init_parameters(Curr_Linked_Ptr_t currPtr) {

		if constexpr(DEBUG_E) type_assert_ptr(Curr_Linked_Ptr_t);

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr size_t N = Curr_Linked_t::N;

		T mean = 0;
		T var = T{1} / T{N};
		var = var > 0 ? var : 1;
		T stdev = std::sqrt(var);

		if constexpr (DEBUG_E > DEBUG0) std::cout << "mean = " << mean << ", stdev = " << stdev << std::endl;

		if constexpr (DEBUG_E > DEBUG0) std::cout << "init random weights DEBUG_E = " << DEBUG_E << std::endl;

		datagenPtr->generate(currPtr->W_begin(), currPtr->W_end(), mean, stdev);

		if constexpr (DEBUG_E >= DEBUG1) {
			datagenPtr->validate(currPtr->W_begin(), currPtr->W_end(), mean, stdev);
		}

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init_patterns(Curr_Linked_Ptr_t currPtr) {

		if constexpr(DEBUG_E) type_assert_ptr(Curr_Linked_Ptr_t);

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;

		if constexpr (DEBUG_E > DEBUG0) std::cout << "... <DEBUG 0> init patterns" << std::endl;

		std::for_each(currPtr->V_begin(),
		              currPtr->V_end(),
		              [&](auto &v) {
			              static int idx = 0;
			              const size_t begin = idx * v.size();
			              const size_t end = (idx + 1) * v.size();
			              idx++;

			              std::transform(
					              datainPtr->cbegin_xtrain() + begin,
					              datainPtr->cbegin_xtrain() + end,
					              v.data(),
					              [&](const auto &x) { return x / T_IN{255}; }
			              );

		              }
		);

		if constexpr(DEBUG_E >= DEBUG1) {
			//datagenPtr->in.validate(feed->begin(), feed->end(), -1, 1);

			T_IN sum = 0;

			std::for_each(currPtr->V_begin(),
			              currPtr->V_end(),
			              [&](auto &v) {
				              sum += std::accumulate(v.data(), v.data() + v.size(), T_IN{0});
			              }
			);

			std::cout << "... <DEBUG 1> : INIT PATTERNS : SUM = " << sum << std::endl;
		}

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init_classifiers(Curr_Linked_Ptr_t currPtr) {

		if constexpr(DEBUG_E) type_assert_ptr(Curr_Linked_Ptr_t);
		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;

		if constexpr (DEBUG_E > DEBUG0)std::cout << "... <DEBUG 0> init patterns" << std::endl;

		std::for_each(currPtr->Z_begin(),
		              currPtr->Z_end(),
		              [&](auto &z) {
			              static int idx = 0;
			              const size_t begin = idx * z.size();
			              const size_t end = (idx + 1) * z.size();
			              idx++;

			              std::copy(datainPtr->cbegin_ztrain() + begin,
			                        datainPtr->cbegin_ztrain() + end,
			                        z.data()
			              );
		              }
		);

		if constexpr(DEBUG_E >= DEBUG1) {

			T_IN sum = 0;

			std::for_each(currPtr->Z_begin(), currPtr->Z_end(),
			              [&](auto &z) {
				              sum += std::accumulate(z.data(), z.data() + z.size(), T_IN{0});
			              }
			);

			std::cout << "... <DEBUG 1> : INIT CLASSIFIERS : SUM = " << sum << std::endl;
		}

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init(Curr_Linked_Ptr_t curr, size_t l) {

		if constexpr(DEBUG_E) type_assert_ptr(Curr_Linked_Ptr_t);

		using Curr_Linked_t = std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr bool HAS_NEXT = Curr_Linked_t::HAS_NEXT;
		constexpr layers_e EID = Curr_Linked_t::EID;

		if constexpr (HAS_NEXT) {

			auto next = curr->get_next();

			if constexpr (DEBUG_E > DEBUG0) std::cout << "FORWARD: curr : " << curr->get_name() << std::endl;

			if constexpr(EID == LAYER_INPUT) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...INPUT" << std::endl;
				init_patterns<DEBUG_E>(curr);

			} else if constexpr (EID == LAYER_HIDDEN) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...HIDDEN" << std::endl;
				init_parameters<DEBUG_E>(curr);
			}

			init<DEBUG_E>(next, l + 1);

			if constexpr (DEBUG_E > DEBUG0)std::cout << "BACKWARD: curr : " << curr->get_name() << std::endl;

			if constexpr(EID == LAYER_INPUT) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...INPUT" << std::endl;
			} else if constexpr (EID == LAYER_HIDDEN) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...HIDDEN" << std::endl;
			}

		} else {

			if constexpr (DEBUG_E > DEBUG0) std::cout << "MIDDLE: is last : " << curr->get_name() << std::endl;
			if constexpr (EID == LAYER_OUTPUT) {
				if constexpr (DEBUG_E > DEBUG0) std::cout << "...OUTPUT" << std::endl;
				init_classifiers<DEBUG_E>(curr);
				init_parameters<DEBUG_E>(curr);
			}
		}

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init() {

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : INIT : START >>>" << std::endl;
			std::cout << "... init input data" << std::endl;

			linkedPtr->print_name();
		}

		init<DEBUG_E>(linkedPtr, 0);

		return this;

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	void Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::print() {

		layersPtr->print(linkedPtr);

	}

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Network<Layers_Ptr_t, Linked_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::deltas(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l) {

		return nullptr;
	}

};

#endif //CPP_EXAMPLE_NNET_HXX
