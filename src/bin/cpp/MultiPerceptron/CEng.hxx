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

    template<typename T, mode_e MODE_E>
    template<debug_e DEBUG_E, class Engine_t, class prev_t, class next_t, class Network_Ptr_t>
    Engine<T, MODE_E>
    ::ParametersEngine<DEBUG_E, Engine_t, prev_t, next_t, Network_Ptr_t>
    ::ParametersEngine(const Engine_t &engine, prev_t prev, next_t next, Network_Ptr_t networkPtr)  :
            engine(engine), prev(prev), next(next), networkPtr(networkPtr) {

        if constexpr (DEBUG_E) {
            type_assert_ptr(prev_t);
            type_assert_ptr(next_t);
            type_assert_ptr(Network_Ptr_t);
        }

    }

    template<typename T, mode_e MODE_E>
    template<debug_e DEBUG_E, class Engine_t, class prev_t, class next_t, class Network_Ptr_t>
    constexpr inline void
    Engine<T, MODE_E>
    ::ParametersEngine<DEBUG_E, Engine_t, prev_t, next_t, Network_Ptr_t>
    ::compute_forward(size_t iter) {

        /* for l in range(1, L):
         *     N, M = curr.shape
         *           curr                              curr           prev             curr
         *     b[l][:M, :mB] = einsum('ij,jm->im', W[l][:M,:N], V[l-1][:N, :mB]) - Theta[l][:M]
         *     V[l][:M, :mB] = g(b[l][:M, :mB])
         *    dV[l][:M, :mB] = dg(V[l][:M, :mB])
         *
*/

        const auto &X = prev->get_V();
        const auto &Theta = next->get_Theta();
        const auto &W = next->get_W();
        auto &b = next->get_b();
        auto &V = next->get_V();
        auto &dV = next->get_dV();

        using W_t= decltype(W);
        using Theta_t = decltype(Theta);
        using X_t = decltype(X);
        using V_t = decltype(V);
        using dV_t = decltype(dV);
        using b_t = decltype(b);

        using compute_forward_t =  typename compute_t:: template Compute_Forward<W_t, Theta_t, X_t, V_t, dV_t, b_t>;
        auto computer = compute_forward_t(W, Theta, X, V, dV, b);
        //auto computer = typename compute_t::Compute_Forward(W, Theta, X, V, dV, b);

        if constexpr(PREV_EID == NNet::LAYER_INPUT) {

            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[INPUT FEED]" << std::endl;
                prev->print_name();
            }
            const auto &idx = networkPtr->get_idx();

            computer.template compute_local_fields(idx, iter);

        } else {

            computer.compute_local_fields();

        }

        if constexpr (DEBUG_E > DEBUG0) {
            std::cout << "[FORWARD] : ... N=" << N << " M=" << M << std::endl;
            prev->print_name();
            next->print_name();
        }

        computer.template compute_activations<T_next>();
    }

    template<typename T, mode_e MODE_E>
    template<debug_e DEBUG_E, class Engine_t, class prev_t, class next_t, class Network_Ptr_t>
    constexpr inline void
    Engine<T, MODE_E>
    ::ParametersEngine<DEBUG_E, Engine_t, prev_t, next_t, Network_Ptr_t>
    ::compute_backward(size_t iter) {

        const auto &W = next->get_W();

        const auto &V = next->get_V();
        const auto &dV = next->get_dV();
        const auto &dX = prev->get_dV();

        auto &DeltaV = next->get_Delta();
        auto &DeltaX = prev->get_Delta();

        using DeltaV_t = decltype(DeltaV);
        using DeltaX_t = decltype(DeltaX);

        using compute_backward_t = typename compute_t::template Compute_Backward<DeltaV_t, DeltaX_t>;
        auto computer = compute_backward_t(DeltaV, DeltaX);

        if constexpr (NEXT_EID == NNet::LAYER_OUTPUT) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[COMPUTE DELTAS BACKWARD FROM OUTPUT] : ..." << std::endl;
                next->print_name();
                prev->print_name();
            }

            const auto &idx = networkPtr->get_idx();
            const auto &Z = next->get_Z();

            computer.template compute_deltas(Z, V, dV, idx, iter);

        } else if constexpr (NEXT_EID == NNet::LAYER_HIDDEN) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[COMPUTE DELTAS BACKWARD FROM HIDDEN] : ..." << std::endl;
                next->print_name();
                prev->print_name();
            }

            computer.template compute_deltas(W, dX);

        }

    }

    template<typename T, mode_e MODE_E>
    template<debug_e DEBUG_E, class Engine_t, class prev_t, class next_t, class Network_Ptr_t>
    constexpr inline void
    Engine<T, MODE_E>
    ::ParametersEngine<DEBUG_E, Engine_t, prev_t, next_t, Network_Ptr_t>
    ::compute_update(size_t iter) {

        const auto &X = prev->get_V();

        const auto &Delta = next->get_Delta();

        auto &dTheta = next->get_zero_dTheta();
        auto &dW = next->get_zero_dW();

        auto &Theta = next->get_Theta();
        auto &W = next->get_W();

        if constexpr (DEBUG_E > DEBUG0) {
            std::cout << "[UPDATE ] : ... N=" << N << " M=" << M << std::endl;
            next->print_name();

        }

        using W_t = decltype(W);
        using Delta_t = decltype(Delta);
        using X_t = decltype(X);
        using dW_t = decltype(dW);
        using Theta_t = decltype(Theta);
        using dTheta_t = decltype(dTheta);

        using compute_update_t = typename compute_t::template Compute_Update<X_t, Delta_t, W_t, dW_t, Theta_t, dTheta_t>;
        auto computer = compute_update_t(X, Delta, W, dW, Theta, dTheta);

        if constexpr(PREV_EID == NNet::LAYER_INPUT) {
            const auto &idx = networkPtr->get_idx();

            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[FIRST HIDDEN LAYER UPDATE]" << std::endl;
                prev->print_name();
            }

            computer.template compute_weights_update(idx, iter);

        } else {

            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[NEXT HIDDEN LAYER UPDATE]" << std::endl;
                prev->print_name();
            }

            computer.compute_weights_update();

        }

        const auto Eta = engine.get_eta();

        computer.compute_bias_update();
        computer.apply_update(Eta);

    }

}



