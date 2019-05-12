//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_NNET_HXX
#define CPP_EXAMPLE_NNET_HXX

#include "NNet.hpp"

#include "CEng.hpp"

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

		if constexpr (not W_ON_STACK) {

			W = W_t(M, N);
			dW = W_t(M, N);

		}

		std::cout << "... <INFO> Layer Hidden : weights [W size : " << W.size() << ", dW size : " << dW.size() << "]" << std::endl;

		if constexpr  (not V_ON_STACK) {

			std::transform(Delta.cbegin(), Delta.cend(), Delta.begin(), [&](const auto &) { return Delta_t(M, 1); });

			std::transform(dV.cbegin(), dV.cend(), dV.begin(), [&](const auto &) { return V_t(M, 1); });
			std::transform(V.cbegin(), V.cend(), V.begin(), [&](const auto &) { return V_t(M, 1); });
			std::transform(b.cbegin(), b.cend(), b.begin(), [&](const auto &) { return V_t(M, 1); });

		}

		std::cout << "... <INFO> Layer Hidden : excitations [V size : " << V.size() << ", dV size : " << dV.size() << "]" << std::endl;
		std::cout << "... <INFO> Layer Hidden : local field [b size : " << b.size() << "]" << std::endl;
		std::cout << "... <INFO> Layer Hidden : deltas      [d size : " << Delta.size() << "]" << std::endl;

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

		if constexpr (not ON_STACK) {
			std::transform(Z.cbegin(), Z.cend(), Z.begin(), [&](const auto &) { return Z_t(N, 1); });
		}

		auto allSizes = std::accumulate(Z.begin(), Z.end(), size_t{0}, [&](const size_t i, const auto &z) { return i + z.size(); });

		std::cout << "... <INFO> Layer Output : classifiers [Z size : " << Z.size() << "]" << std::endl;
		std::cout << "... <INFO> Layer Output : classifiers [Z[1.size + ... + N.size] size : " << allSizes << "]" << std::endl;

	}

	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	auto LayerOutput<T, B, mB, N, M>::Z_begin() {

		return Z.begin();
	}

	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	auto LayerOutput<T, B, mB, N, M>::Z_end() {

		return Z.end();
	}

	template<typename T, size_t B, size_t mB, size_t N>
	LayerInput<T, B, mB, N>::LayerInput(size_t id) :
			LayerBase(id, 0, N, NAME) {

		if constexpr (not ON_STACK) {

			std::transform(V.cbegin(), V.cend(), V.begin(), [&](const auto &) { return V_t(N, 1); });

		}

		auto allSizes = std::accumulate(V.begin(), V.end(), size_t{0}, [&](const size_t i, const auto &x) { return i + x.size(); });
		if (V.empty()) {
			std::cout << "... <ERROR> : Layer Input not allocated [V]..." << std::endl;
		} else {
			std::cout << "... <INFO> : Layer Input allocated [V size : " << V.size() << "]..." << std::endl;
			std::cout << "... <INFO> : Layer Input allocated [V[1.size + ... + N.size] size : " << allSizes << "]" << std::endl;
		}

	}

	template<typename T, size_t B, size_t mB, size_t N>
	void LayerInput<T, B, mB, N>::print() {

		LayerBase::print();
	}

	template<typename T, size_t B, size_t mB, size_t N>
	auto LayerInput<T, B, mB, N>::V_begin() {

		return V.begin();
	}

	template<typename T, size_t B, size_t mB, size_t N>
	auto LayerInput<T, B, mB, N>::V_end() {

		return V.end();
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

		obj_assert_ptr(next);

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

		obj_assert_ptr(next);

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

		obj_assert_ptr(next);

		return new Linked<LayerInput<T_IN, B, mB, I>, decltype(next)>(next, 0);

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::print(const Linked_Ptr_t layers) {

		type_assert_ptr(Linked_Ptr_t);

		layers->print();

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

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::Network(const Layers_t &layers, const Linked_Ptr_t &linkedPtr, const Engine_Ptr_t &enginePtr, const Datain_Ptr_t &datainPtr, const Datagen_Ptr_t &datagenPtr) :
			layers(layers), linkedPtr(linkedPtr), enginePtr(enginePtr), datainPtr(datainPtr), datagenPtr(datagenPtr), IDX(B) {

//		type_assert_ptr(Linked_Ptr_t);
//		type_assert_ptr(Engine_Ptr_t);
//		type_assert_ptr(Datain_Ptr_t);
//		type_assert_ptr(Datagen_Ptr_t);

		std::iota(IDX.begin(), IDX.end(), size_t{0});
	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	void Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::shuffle_idx() {

		static std::random_device randEng;
		std::mt19937 g(randEng());
		std::shuffle(IDX.begin(), IDX.end(), g);

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	const void Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::shuffle_idx() const {

		static std::random_device randEng;
		std::mt19937 g(randEng());
		std::shuffle(IDX.begin(), IDX.end(), g);

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	const auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::get_next_mb() const {

		static size_t k = 0;
		size_t begin = k * mB;
		size_t end = (k + 1) * mB;
		k = (k + 1) % (B / mB);

		return std::tuple(k, begin, end);

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, compute_e COMPUTE_E, class Curr_Linked_Ptr_t>
	void Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::compute(Curr_Linked_Ptr_t curr, size_t l, size_t epoch, size_t iter) {

		using Curr_Linked_t = std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr bool HAS_NEXT = Curr_Linked_t::HAS_NEXT;
		constexpr bool HAS_PREV = Curr_Linked_t::HAS_PREV;
		auto next = curr->get_next();

		if constexpr (HAS_NEXT) {

			using curr_t = decltype(curr);
			using next_t = decltype(next);
			using this_t = decltype(this);

			using engine_t = typename Engine_Ptr_t::template ParametersEngine<DEBUG_E, Engine_Ptr_t, curr_t, next_t, this_t>;
			static auto engine = engine_t(enginePtr, curr, next, this);

			if constexpr (COMPUTE_E == FORWARD_E) {
				engine.compute_forward(iter);
				//compute<DEBUG_E, COMPUTE_E>(curr, next, l);
				compute < DEBUG_E, COMPUTE_E > (next, l + 1, epoch, iter);
			} else if constexpr (COMPUTE_E == UPDATE_E) {
				engine.compute_update(iter);
				//compute<DEBUG_E, COMPUTE_E>(curr, next, l); //next, curr

				compute < DEBUG_E, COMPUTE_E > (next, l + 1, epoch, iter);

			} else if constexpr(COMPUTE_E == BACKWARD_E) {
				compute < DEBUG_E, COMPUTE_E > (next, l + 1, epoch, iter);
				if constexpr (HAS_PREV) {
					engine.compute_backward(iter);
				}
			}

		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E>
	void Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::compute(size_t epochs) {

		const size_t EPOCHS = epochs;

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : COMPUTE : START [DEBUG] >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
		}

		{
			TIME_START
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[RESAMPLE INPUT]" << std::endl;
				linkedPtr->print_name();
			}

			shuffle_idx();

			/* 1. Resample mB minibatch from B images xTrain
			 * 2. Match mB minibatch from B labels zTrain
			 */

			compute < DEBUG_E, FORWARD_E > (linkedPtr, 0, 0, 0);
			compute < DEBUG_E, BACKWARD_E > (linkedPtr, 0, 0, 0);
			compute < DEBUG_E, UPDATE_E > (linkedPtr, 0, 0, 0);

			TIME_CHECK
		}

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : COMPUTE : START [EPOCHS : " << epochs << ", ITER : " << ITER << "] >>>"
			          << std::endl;
			type_assert_ptr(Linked_Ptr_t);
		}

		size_t epoch = 0;
		TIME_START
		do {

			std::cout << "...epochs : " << epochs << std::endl;

			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[RESAMPLE INPUT]" << std::endl;
				linkedPtr->print_name();
			}

			shuffle_idx();

			/* 1. Resample mB minibatch from B images xTrain
			 * 2. Match mB minibatch from B labels zTrain
			 */

			size_t iterations = ITER;
			do {
				if (iterations % (ITER / 10) == 0) std::cout << "...iterations : " << iterations << std::endl;
				compute < DEBUG0, FORWARD_E > (linkedPtr, 0, EPOCHS - epoch, ITER - iterations);
				compute < DEBUG0, BACKWARD_E > (linkedPtr, 0, EPOCHS - epoch, ITER - iterations);
				compute < DEBUG0, UPDATE_E > (linkedPtr, 0, EPOCHS - epoch, ITER - iterations);
			} while (--iterations);
		} while (--epochs);

		TIME_CHECK
	}

	template<class Layers_t, class Linked_Ptr, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init_parameters(Curr_Linked_Ptr_t currLinkedPtr) {

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr size_t N = Curr_Linked_t::N;

		T mean = 0;
		T var = T{1} / T{N};
		var = var > 0 ? var : 1;
		T stdev = std::sqrt(var);

		if constexpr (DEBUG_E > DEBUG0) std::cout << "mean = " << mean << ", stdev = " << stdev << std::endl;

		if constexpr (DEBUG_E > DEBUG0) std::cout << "init random weights DEBUG_E = " << DEBUG_E << std::endl;

		if constexpr (DEBUG_E >= DEBUG0) {
			datagenPtr.generate(currLinkedPtr->W_begin(), currLinkedPtr->W_end(), mean, stdev);
		}

		if constexpr (DEBUG_E >= DEBUG1) {
			datagenPtr.validate(currLinkedPtr->W_begin(), currLinkedPtr->W_end(), mean, stdev);
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init_patterns(Curr_Linked_Ptr_t currLinkedPtr) {

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;

		if constexpr(DEBUG_E >= DEBUG0) {

			if constexpr (DEBUG_E > DEBUG0) std::cout << "... <DEBUG 0> init patterns" << std::endl;

			std::accumulate(
					currLinkedPtr->V_begin(),
					currLinkedPtr->V_end(),
					size_t{0},
					[&](size_t idx, auto &v) {
						const size_t begin = idx * v.size();
						const size_t end = (idx + 1) * v.size();
						std::transform(
								datainPtr.cbegin_xtrain() + begin,
								datainPtr.cbegin_xtrain() + end,
								v.data(),
								[&](const auto &x) -> T_IN { return static_cast<T_IN>(x) / T_IN{255}; }

						);
						return idx++;
					}
			);
		}

		if constexpr(DEBUG_E >= DEBUG1) {
			//datagenPtr.in.validate(feed->begin(), feed->end(), -1, 1);

			T_IN sum = 0;

			std::accumulate(
					currLinkedPtr->V_begin(),
					currLinkedPtr->V_end(),
					T_IN{0},
					[&](size_t idx, auto &v) {
						sum += std::accumulate(v.data(), v.data() + v.size(), T_IN{0});
						return idx++;
					});

			std::cout << "... <DEBUG 1> : INIT PATTERNS : SUM = " << sum << std::endl;
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init_classifiers(Curr_Linked_Ptr_t currLinkedPtr) {

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;

		if constexpr(DEBUG_E >= DEBUG0) {
			if constexpr (DEBUG_E > DEBUG0)std::cout << "... <DEBUG 0> init patterns" << std::endl;

			std::accumulate(
					currLinkedPtr->Z_begin(),
					currLinkedPtr->Z_end(),
					size_t{0},
					[&](size_t idx, auto &z) {
						const size_t begin = idx * z.size();
						const size_t end = (idx + 1) * z.size();
						std::copy(datainPtr.cbegin_ztrain() + begin, datainPtr.cbegin_ztrain() + end, z.data());
						return idx++;
					}
			);
		}

		if constexpr(DEBUG_E >= DEBUG1) {

			T_IN sum = 0;

			std::accumulate(
					currLinkedPtr->Z_begin(),
					currLinkedPtr->Z_end(),
					T_IN{0},
					[&](size_t idx, auto &z) {
						sum += std::accumulate(z.data(), z.data() + z.size(), T_IN{0});
						return idx++;
					});

			std::cout << "... <DEBUG 1> : INIT CLASSIFIERS : SUM = " << sum << std::endl;
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init(Curr_Linked_Ptr_t curr, size_t l) {

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

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	template<debug_e DEBUG_E>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::init() {

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : INIT : START >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
			//type_assert_ptr(Datagen_Ptr_t);
			std::cout << "... init input data" << std::endl;

			linkedPtr->print_name();
		}

		init<DEBUG_E>(linkedPtr, 0);

		return this;

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	void Network<Layers_t, Linked_Ptr_t, Engine_Ptr_t, Datain_Ptr_t, Datagen_Ptr_t>
	::print() {

		layers.print(linkedPtr);

	}

};

#endif //CPP_EXAMPLE_NNET_HXX
