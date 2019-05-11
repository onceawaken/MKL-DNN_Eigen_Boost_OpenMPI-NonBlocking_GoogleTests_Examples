//
// Created by egrzrbr on 2019-04-06.
//


#include "CEng.hpp"

namespace CEng {

	template<typename Iterator>
	typename std::iterator_traits<Iterator>::value_type
	sum(Iterator begin, Iterator end) {

		using value_type = typename std::iterator_traits<Iterator>::value_type;
		value_type s = value_type();
		for (Iterator it = begin; it != end; it++) {
			s += *it;
		}
		return s;
	}

	template<typename T, class Matrix>
	constexpr inline auto sigmoid(const Matrix &M) {
		T _1 = T{1};
		return (_1 + (-M).array().exp()).inverse().matrix();
	}

	template<typename T, class Matrix>
	constexpr inline auto dsigmoid(const Matrix &S /*S = sigmoid(M)*/) {
		T _1 = T{1};
		return S.array() * (_1 - S.array());

	}

	template<typename T, class Matrix>
	constexpr inline auto soft_max(const Matrix &M) {
		T max = M.maxCoeff();
		return (M.array() - (Eigen::log((M.array() - max).exp().sum()) + max)).exp();

	}

	template<typename T, class Matrix>
	constexpr inline auto log_soft_max(const Matrix &M) {
		T max = M.maxCoeff();
		T sum = (M.array() - max).exp().sum();
		T logZ = Eigen::log(sum) + max;
		return M.array() - logZ;
	}

	template<typename T, class Matrix>
	constexpr inline auto log_soft_max(const Matrix &M, T &logZ) {
		T max = M.maxCoeff();
		T sum = (M.array() - max).exp().sum();
		logZ = Eigen::log(sum) + max;
		return M.array() - logZ;
	}

	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
	Engine
	::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t, Network_Ptr_t>
	::InnerEngine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, Network_Ptr_t networkPtr)  :
			prevLinkedPtr(prevLinkedPtr), nextLinkedPtr(nextLinkedPtr), networkPtr(networkPtr) {

		if constexpr (DEBUG_E) {
			type_assert_ptr(Prev_Linked_Ptr_t);
			type_assert_ptr(Next_Linked_Ptr_t);
			type_assert_ptr(Network_Ptr_t);
		}

	}

	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
	void Engine
	::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t, Network_Ptr_t>
	::compute_forward(size_t epoch, size_t iter) {

		/* for l in range(1, L):
		 *     N, M = curr.shape
		 *           curr                              curr           prev             curr
		 *     b[l][:M, :mB] = einsum('ij,jm->im', W[l][:M,:N], V[l-1][:N, :mB]) - Theta[l][:M]
		 *     V[l][:M, :mB] = g(b[l][:M, :mB])
		 *    dV[l][:M, :mB] = dg(V[l][:M, :mB])
		 *
*/

		const auto &X = prevLinkedPtr->get_V();

		const auto &Theta = nextLinkedPtr->get_Theta();
		const auto &W = nextLinkedPtr->get_W();
		auto &b = nextLinkedPtr->get_b();
		auto &V = nextLinkedPtr->get_V();
		auto &dV = nextLinkedPtr->get_dV();

		if constexpr(PREV_EID == NNet::LAYER_INPUT) {
			const auto &idx = networkPtr->get_idx();

			const size_t x_begin = iter * mB;
			const size_t x_end = (iter + 1) * mB;

			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[INPUT FEED]" << std::endl;
				prevLinkedPtr->print_name();
			}

			std::transform(idx.cbegin() + x_begin, idx.cbegin() + x_end, b.begin(),
			               [&](const size_t i) { return W * X[i].matrix() - Theta; }
			);

		} else {
			std::transform(X.cbegin(), X.cend(), b.begin(), [&](const auto &x) { return W * x.matrix() - Theta; });
		}

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "[FORWARD] : ... N=" << N << " M=" << M << std::endl;
			prevLinkedPtr->print_name();
			nextLinkedPtr->print_name();

		}

//		//sigmoid

		std::transform(b.cbegin(), b.cend(), V.begin(), [&](const auto &B) { return sigmoid<T_next>(B); });
		std::transform(V.cbegin(), V.cend(), dV.begin(), [&](const auto &S) { return dsigmoid<T_next>(S); });

	}

	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
	void
	Engine::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t, Network_Ptr_t>
	::compute_backward(size_t epoch, size_t iter) {

		const auto &Theta = nextLinkedPtr->get_Theta();
		const auto &W = nextLinkedPtr->get_W();

		const auto &V_up = nextLinkedPtr->get_V();
		const auto &V_dn = prevLinkedPtr->get_V();

		const auto &dV_up = nextLinkedPtr->get_dV();
		const auto &dV_dn = prevLinkedPtr->get_dV();

		auto &DeltaUp = nextLinkedPtr->get_Delta();
		auto &DeltaDn = prevLinkedPtr->get_Delta();

		const size_t x_begin = iter * mB;
		const size_t x_end = (iter + 1) * mB;

		if constexpr (NEXT_EID == NNet::LAYER_OUTPUT) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[COMPUTE DELTAS BACKWARD FROM OUTPUT] : ..." << std::endl;
				nextLinkedPtr->print_name();
				prevLinkedPtr->print_name();
			}

			const auto &idx = networkPtr->get_idx();

			const auto &Z = nextLinkedPtr->get_Z();

			std::transform(V_up.cbegin(), V_up.cend(), idx.cbegin() + x_begin, DeltaUp.begin(),
			               [&](const auto &V, const auto &k) {
				               return Z[k] - V;
			               }
			);

		} else if constexpr (NEXT_EID == NNet::LAYER_HIDDEN) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[COMPUTE DELTAS BACKWARD FROM HIDDEN] : ..." << std::endl;
				nextLinkedPtr->print_name();
				prevLinkedPtr->print_name();
			}

			std::transform(DeltaUp.cbegin(), DeltaUp.cend(), DeltaDn.begin(),
			               [&](const auto &delta) {
				               return W.transpose() * delta.matrix();
			               }
			);
		}

		std::transform(dV_dn.cbegin(), dV_dn.cend(), DeltaDn.cbegin(), DeltaDn.begin(),
		               [&](const auto &dV, const auto &delta) {
			               return dV * delta;
		               }
		);

	}

	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t, class Network_Ptr_t>
	void Engine::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t, Network_Ptr_t>
	::compute_update(size_t epoch, size_t iter) {

		const auto &V = prevLinkedPtr->get_V();

		const auto &Delta = nextLinkedPtr->get_Delta();

		auto &dTheta = nextLinkedPtr->get_zero_dTheta();
		auto &dW = nextLinkedPtr->get_zero_dW();

		auto &Theta = nextLinkedPtr->get_Theta();
		auto &W = nextLinkedPtr->get_W();

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "[UPDATE ] : ... N=" << N << " M=" << M << std::endl;
			nextLinkedPtr->print_name();
			std::cout << "delta = " << Delta[0].rows() << "," << Delta[0].cols() << std::endl;
			std::cout << "V = " << V[0].rows() << "," << V[0].cols() << std::endl;
			std::cout << "dW = " << dW.rows() << "," << dW.cols() << std::endl;
		}

		if constexpr(PREV_EID == NNet::LAYER_INPUT) {
			const auto &idx = networkPtr->get_idx();

			const size_t x_begin = iter * mB;
			const size_t x_end = (iter + 1) * mB;

			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[FIRST HIDDEN LAYER UPDATE]" << std::endl;
				prevLinkedPtr->print_name();
			}

			//This sequential crap take +100s, without it training is 10-30s
			dW = std::inner_product(idx.cbegin() + x_begin, idx.cbegin() + x_end, Delta.cbegin(), dW,
			                        [&](const auto &dW_curr, const auto &dW_new) { return dW_curr + dW_new; },
			                        [&](const size_t i, const auto &delta) { return delta.matrix() * V[i].matrix().transpose(); }
			);

		} else {

			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[NEXT HIDDEN LAYER UPDATE]" << std::endl;
				prevLinkedPtr->print_name();
			}


//			for (int i = 0; i < mB; i++) {
//				dW += Delta[i].matrix() * V[i].matrix().transpose();
//			}
//			dW = std::transform_reduce(pstl::execution::par, V.cbegin(), V.cend(), Delta.cbegin(), dW,
//			                        [&](const auto &dW_curr, const auto &dW_new) { return dW_curr + dW_new; },
//			                        [&](const auto &v, const auto &delta) { return delta.matrix() * v.matrix().transpose(); }
//			);

			dW = std::inner_product(V.cbegin(), V.cend(), Delta.cbegin(), dW,
			                        [&](const auto &dW_curr, const auto &dW_new) { return dW_curr + dW_new; },
			                        [&](const auto &v, const auto &delta) { return delta.matrix() * v.matrix().transpose(); }
			);

		}

		dTheta = std::accumulate(Delta.cbegin(), Delta.cend(), dTheta,
		                         [&](const auto &dtheta, const auto &delta) { return dtheta + delta.matrix(); });

		W += dW; // * eta        ;
		dTheta -= dTheta; //*eta;

	}

}



