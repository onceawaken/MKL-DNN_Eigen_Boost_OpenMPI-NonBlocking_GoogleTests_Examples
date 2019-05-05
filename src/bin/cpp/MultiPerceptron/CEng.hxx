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


    template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
    Engine
    ::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>
    ::InnerEngine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr)  :
            prevLinkedPtr(prevLinkedPtr), nextLinkedPtr(nextLinkedPtr) {

        if constexpr (DEBUG_E) {
            type_assert_ptr(Prev_Linked_Ptr_t);
            type_assert_ptr(Next_Linked_Ptr_t);
        }

    }

    template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
    void Engine
    ::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>
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
            const auto &idx = prevLinkedPtr->get_idx();

            const size_t x_begin = iter * mB;
            const size_t x_end = (iter + 1) * mB;


            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[INPUT FEED]" << std::endl;
                prevLinkedPtr->print_name();
            }

            std::transform(idx.cbegin() + x_begin, idx.cbegin() + x_end, b.begin(),
                           [&](const size_t i) { return W * X[i] - Theta; }
            );


        } else {
            std::transform(X.cbegin(), X.cend(), b.begin(), [&](const auto &x) { return W * x - Theta; });
        }

        if constexpr (DEBUG_E > DEBUG0) {
            std::cout << "[FORWARD] : ... N=" << N << " M=" << M << std::endl;
            prevLinkedPtr->print_name();
            nextLinkedPtr->print_name();

        }


        if constexpr (NEXT_EID == NNet::LAYER_OUTPUT) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[COMPUTE DELTAS] : ... N=" << N << " M=" << M << std::endl;
                nextLinkedPtr->print_name();

                auto &Delta = nextLinkedPtr->get_Delta();


            }

        }


    }

    template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
    void
    Engine::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>::compute_backward(size_t epoch, size_t iter) {

//        const auto &X = prevLinkedPtr->get_V();
//
//        const auto &Theta = nextLinkedPtr->get_Theta();
//        const auto &W = nextLinkedPtr->get_W();
//        auto &b = nextLinkedPtr->get_b();
//        auto &V = nextLinkedPtr->get_V();
//        auto &dV = nextLinkedPtr->get_dV();
//
//        const auto & dVprev = prevLinkedPtr->get_dV();
//
//        const auto & Wnext = nextLinkedPtr->get_W();
//        auto & Wprev = prevLinkedPtr->get_W();
//
//
//        const auto & DeltaNext = nextLinkedPtr->get_Delta();
//        auto & DeltaPrev = prevLinkedPtr->get_Delta();

        if constexpr (PREV_EID == NNet::LAYER_OUTPUT) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[BACKWARD FROM OUTPUT] : ..." << std::endl;
                nextLinkedPtr->print_name();
                prevLinkedPtr->print_name();

            }


        } else if constexpr (PREV_EID == NNet::LAYER_HIDDEN) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[BACKWARD FROM HIDDEN] : ..." << std::endl;
                nextLinkedPtr->print_name();
                prevLinkedPtr->print_name();


            }

        }

    }

    template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
    void Engine::InnerEngine<DEBUG_E, Prev_Linked_Ptr_t, Next_Linked_Ptr_t>::compute_update(size_t epoch, size_t iter) {
        //        const auto &X = prevLinkedPtr->get_V();
//
//        const auto &Theta = nextLinkedPtr->get_Theta();
//        const auto &W = nextLinkedPtr->get_W();
//        auto &b = nextLinkedPtr->get_b();
//        auto &V = nextLinkedPtr->get_V();
//        auto &dV = nextLinkedPtr->get_dV();
//
//        const auto & dVprev = prevLinkedPtr->get_dV();
//
//        const auto & Wnext = nextLinkedPtr->get_W();
//        auto & Wprev = prevLinkedPtr->get_W();
//
//
//        const auto & DeltaNext = nextLinkedPtr->get_Delta();
//        auto & DeltaPrev = prevLinkedPtr->get_Delta();

        if constexpr (NEXT_EID == NNet::LAYER_HIDDEN) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[UPDATE FROM HIDDEN] : ... N=" << N << " M=" << M << std::endl;
                prevLinkedPtr->print_name();
                nextLinkedPtr->print_name();

            }

        } else if constexpr (NEXT_EID == NNet::LAYER_OUTPUT) {
            if constexpr (DEBUG_E > DEBUG0) {
                std::cout << "[UPDATE AT OUTPUT] : ... N=" << N << " M=" << M << std::endl;
                prevLinkedPtr->print_name();
                nextLinkedPtr->print_name();

                auto &Delta = nextLinkedPtr->get_Delta();

            }

        }


    }


}

