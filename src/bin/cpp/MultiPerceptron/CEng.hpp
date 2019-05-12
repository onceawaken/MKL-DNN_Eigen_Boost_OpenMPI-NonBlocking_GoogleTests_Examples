//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_CENG_HPP
#define CPP_EXAMPLE_CENG_HPP

namespace CEng {

    template<typename T, class Matrix>
    constexpr inline auto sigmoid(const Matrix &M);

    template<typename T, class Matrix>
    constexpr inline auto dsigmoid(const Matrix &S /*S = sigmoid(M)*/);

    template<typename T, class Matrix>
    constexpr inline auto soft_max(const Matrix &M);

    template<typename T, class Matrix>
    constexpr inline auto log_soft_max(const Matrix &M);

    template<typename T, class Matrix>
    constexpr inline auto log_soft_max(const Matrix &M, T &logZ);

    enum mode_e {
        DUMMY_E = 0,
        SEQUENTIAL_E = 1,
        MTHREADED_E = 2,
        MPROCESSING_E = 4,
    };

    class Compute_Dummy {
      public:
        template<class W_t, class Theta_t, class X_t, class V_t, class dV_t, class b_t>
        class Compute_Forward {

            const Theta_t &Theta;
            const W_t &W;
            const X_t &X;

            dV_t &dV;
            V_t &V;
            b_t &b;

          public:

            Compute_Forward(const W_t &W, const Theta_t &Theta, const X_t &X, V_t &V, dV_t &dV, b_t &b)
                    : W(W), Theta(Theta), X(X), V(V), dV(dV), b(b) {

            }

            template<class idx_t>
            constexpr inline void compute_local_fields(const idx_t &idx, const size_t iter) {
                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            constexpr inline void compute_local_fields() {

            }

            template<typename T>
            constexpr inline void compute_activations() {
            }

        };

        template<class DeltaV_t, class DeltaX_t>
        class Compute_Backward {

            DeltaX_t &DeltaX;
            DeltaV_t &DeltaV;

          public:

            Compute_Backward(DeltaV_t &DeltaV, DeltaX_t &DeltaX) :
                    DeltaX(DeltaX), DeltaV(DeltaV) {

            }

            template<class Z_t, class V_t, class dV_t, class idx_t>
            constexpr inline void
            compute_deltas(const Z_t &Z, const V_t &V, const dV_t &dV, const idx_t &idx, const size_t iter) {

                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            template<class W_t, class dX_t>
            constexpr inline void compute_deltas(const W_t &W, const dX_t &dX) {

            }

        };

        template<class X_t, class Delta_t, class W_t, class dW_t, class Theta_t, class dTheta_t>
        class Compute_Update {

            W_t &W;
            dW_t &dW;
            Theta_t &Theta;
            dTheta_t &dTheta;

            const X_t &X;
            const Delta_t &Delta;

          public:

            Compute_Update(const X_t &X, const Delta_t &Delta, W_t &W, dW_t &dW, Theta_t &Theta, dTheta_t &dTheta) :
                    X(X), Delta(Delta), W(W), dW(dW), Theta(Theta), dTheta(dTheta) {

            }

            template<class idx_t>
            constexpr inline void compute_weights_update(const idx_t &idx, const size_t iter) {
                const size_t mB = Delta.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;

            }

            constexpr inline void compute_weights_update() {

            }

            constexpr inline void compute_bias_update() {

            }

            template<typename T>
            constexpr inline void apply_update(const T eta) {

            }

        };

    };

    class Compute_Sequential {
      public:
        template<class W_t, class Theta_t, class X_t, class V_t, class dV_t, class b_t>
        class Compute_Forward {

            const Theta_t &Theta;
            const W_t &W;
            const X_t &X;

            const dV_t &dV;
            const V_t &V;
            const b_t &b;

          public:

             Compute_Forward(const W_t &W, const Theta_t &Theta, const X_t &X, const V_t &V, const dV_t &dV,
                            const b_t &b)
                    : W(W), Theta(Theta), X(X), V(V), dV(dV), b(b) {

            }

            template<class idx_t>
             inline void compute_local_fields(const idx_t &idx, const size_t iter) {
                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;

                std::transform(idx.cbegin() + x_begin, idx.cbegin() + x_end, b.begin(),
                               [&](const size_t i) {

                                   return W * X[i].matrix() - Theta;
                               }
                );
            }

             inline void compute_local_fields() {

                std::transform(X.cbegin(), X.cend(), b.begin(), [&](const auto &x) { return W * x.matrix() - Theta; });
            }

            template<typename T>
             inline void compute_activations() {
                std::transform(b.cbegin(), b.cend(), V.begin(), [&](const auto &B) { return sigmoid<T>(B); });
                std::transform(V.cbegin(), V.cend(), dV.begin(), [&](const auto &S) { return dsigmoid<T>(S); });
            }

        };

        template<class DeltaV_t, class DeltaX_t>
        class Compute_Backward {

            const DeltaX_t &DeltaX;
            const DeltaV_t &DeltaV;

          public:

            Compute_Backward(const DeltaV_t &DeltaV, const DeltaX_t &DeltaX) :
                    DeltaX(DeltaX), DeltaV(DeltaV) {

            }

            template<class Z_t, class V_t, class dV_t, class idx_t>
             inline void
            compute_deltas(const Z_t &Z, const V_t &V, const dV_t &dV, const idx_t &idx, const size_t iter) {

                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;

                std::transform(V.cbegin(), V.cend(), idx.cbegin() + x_begin, DeltaV.begin(),
                               [&](const auto &v, const auto &k) {
                                   return Z[k] - v;
                               }
                );

                std::transform(DeltaV.cbegin(), DeltaV.cend(), dV.begin(), DeltaV.begin(),
                               [&](const auto &deltav, const auto &dv) {
                                   return dv * deltav;
                               }
                );

            }

            template<class W_t, class dX_t>
             inline void compute_deltas(const W_t &W, const dX_t &dX) {
                std::transform(DeltaV.cbegin(), DeltaV.cend(), DeltaX.begin(),
                               [&](const auto &deltav) {
                                   return W.transpose().lazyProduct(deltav.matrix());
                               }
                );

                std::transform(dX.cbegin(), dX.cend(), DeltaX.cbegin(), DeltaX.begin(),
                               [&](const auto &dx, const auto &deltax) {
                                   return dx * deltax;
                               }
                );
            }

        };

        template<class X_t, class Delta_t, class W_t, class dW_t, class Theta_t, class dTheta_t>
        class Compute_Update {

            const W_t &W;
            const dW_t &dW;
            const Theta_t &Theta;
            const dTheta_t &dTheta;

            const X_t &X;
            const Delta_t &Delta;

          public:

            Compute_Update(const X_t &X, const Delta_t &Delta, const W_t &W, const dW_t &dW, const Theta_t &Theta,const dTheta_t &dTheta) :
                    X(X), Delta(Delta), W(W), dW(dW), Theta(Theta), dTheta(dTheta) {

            }

            template<class idx_t>
             inline void compute_weights_update(const idx_t &idx, const size_t iter) {
                const size_t mB = Delta.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;
                //This sequential crap take 70% time, optimise it!
                dW = std::inner_product(idx.cbegin() + x_begin, idx.cbegin() + x_end, Delta.cbegin(), dW,
                                        [&](const auto &dW_curr, const auto &dW_new) { return dW_curr + dW_new; },
                                        [&](const size_t i, const auto &delta) {
                                            return delta.matrix() * X[i].matrix().transpose();
                                        }
                );
            }

             inline void compute_weights_update() {
//			dW = std::transform_reduce(pstl::execution::par, V.cbegin(), V.cend(), Delta.cbegin(), dW,
//			                        [&](const auto &dW_curr, const auto &dW_new) { return dW_curr + dW_new; },
//			                        [&](const auto &v, const auto &delta) { return delta.matrix() * v.matrix().transpose(); }
//			);

                dW = std::inner_product(X.cbegin(), X.cend(), Delta.cbegin(), dW,
                                        [&](const auto &dW_curr, const auto &dW_new) { return dW_curr + dW_new; },
                                        [&](const auto &x, const auto &delta) {
                                            return delta.matrix() * x.matrix().transpose();
                                        }
                );

                //dW, Delta, X
            }

             inline void compute_bias_update() {
                dTheta = std::accumulate(Delta.cbegin(), Delta.cend(), dTheta,
                                         [&](const auto &dtheta, const auto &delta) {
                                             return dtheta + delta.matrix();
                                         });
            }

            template<typename T>
             inline void apply_update(const T eta) {


                //Delta, dTheta

                W += dW * eta;
                Theta -= dTheta * eta;
            }

        };

    };

    class Compute_Mthreaded {
      public:
        template<class W_t, class Theta_t, class X_t, class V_t, class dV_t, class b_t>
        class Compute_Forward {

            const Theta_t &Theta;
            const W_t &W;
            const X_t &X;

            dV_t &dV;
            V_t &V;
            b_t &b;

          public:

            Compute_Forward(const W_t &W, const Theta_t &Theta, const X_t &X, V_t &V, dV_t &dV, b_t &b)
                    : W(W), Theta(Theta), X(X), V(V), dV(dV), b(b) {

            }

            template<class idx_t>
            constexpr inline void compute_local_fields(const idx_t &idx, const size_t iter) {
                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            constexpr inline void compute_local_fields() {

            }

            template<typename T>
            constexpr inline void compute_activations() {
            }

        };

        template<class DeltaV_t, class DeltaX_t>
        class Compute_Backward {

            DeltaX_t &DeltaX;
            DeltaV_t &DeltaV;

          public:

            Compute_Backward(DeltaV_t &DeltaV, DeltaX_t &DeltaX) :
                    DeltaX(DeltaX), DeltaV(DeltaV) {

            }

            template<class Z_t, class V_t, class dV_t, class idx_t>
            constexpr inline void
            compute_deltas(const Z_t &Z, const V_t &V, const dV_t &dV, const idx_t &idx, const size_t iter) {

                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            template<class W_t, class dX_t>
            constexpr inline void compute_deltas(const W_t &W, const dX_t &dX) {

            }

        };

        template<class X_t, class Delta_t, class W_t, class dW_t, class Theta_t, class dTheta_t>
        class Compute_Update {

            W_t &W;
            dW_t &dW;
            Theta_t &Theta;
            dTheta_t &dTheta;

            const X_t &X;
            const Delta_t &Delta;

          public:

            Compute_Update(const X_t &X, const Delta_t &Delta, W_t &W, dW_t &dW, Theta_t &Theta, dTheta_t &dTheta) :
                    X(X), Delta(Delta), W(W), dW(dW), Theta(Theta), dTheta(dTheta) {

            }

            template<class idx_t>
            constexpr inline void compute_weights_update(const idx_t &idx, const size_t iter) {
                const size_t mB = Delta.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;

            }

            constexpr inline void compute_weights_update() {

            }

            constexpr inline void compute_bias_update() {

            }

            template<typename T>
            constexpr inline void apply_update(const T eta) {

            }

        };

    };

    class Compute_Mprocessing {
      public:
        template<class W_t, class Theta_t, class X_t, class V_t, class dV_t, class b_t>
        class Compute_Forward {

            const Theta_t &Theta;
            const W_t &W;
            const X_t &X;

            dV_t &dV;
            V_t &V;
            b_t &b;

          public:

            Compute_Forward(const W_t &W, const Theta_t &Theta, const X_t &X, V_t &V, dV_t &dV, b_t &b)
                    : W(W), Theta(Theta), X(X), V(V), dV(dV), b(b) {

            }

            template<class idx_t>
            constexpr inline void compute_local_fields(const idx_t &idx, const size_t iter) {
                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            constexpr inline void compute_local_fields() {

            }

            template<typename T>
            constexpr inline void compute_activations() {
            }

        };

        template<class DeltaV_t, class DeltaX_t>
        class Compute_Backward {

            DeltaX_t &DeltaX;
            DeltaV_t &DeltaV;

          public:

            Compute_Backward(DeltaV_t &DeltaV, DeltaX_t &DeltaX) :
                    DeltaX(DeltaX), DeltaV(DeltaV) {

            }

            template<class Z_t, class V_t, class dV_t, class idx_t>
            constexpr inline void
            compute_deltas(const Z_t &Z, const V_t &V, const dV_t &dV, const idx_t &idx, const size_t iter) {

                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            template<class W_t, class dX_t>
            constexpr inline void compute_deltas(const W_t &W, const dX_t &dX) {

            }

        };

        template<class X_t, class Delta_t, class W_t, class dW_t, class Theta_t, class dTheta_t>
        class Compute_Update {

            W_t &W;
            dW_t &dW;
            Theta_t &Theta;
            dTheta_t &dTheta;

            const X_t &X;
            const Delta_t &Delta;

          public:

            Compute_Update(const X_t &X, const Delta_t &Delta, W_t &W, dW_t &dW, Theta_t &Theta, dTheta_t &dTheta) :
                    X(X), Delta(Delta), W(W), dW(dW), Theta(Theta), dTheta(dTheta) {

            }

            template<class idx_t>
            constexpr inline void compute_weights_update(const idx_t &idx, const size_t iter) {
                const size_t mB = Delta.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;

            }

            constexpr inline void compute_weights_update() {

            }

            constexpr inline void compute_bias_update() {

            }

            template<typename T>
            constexpr inline void apply_update(const T eta) {

            }

        };

    };

    class Compute_Mthreaded_Mprocessing {
      public:
        template<class W_t, class Theta_t, class X_t, class V_t, class dV_t, class b_t>
        class Compute_Forward {

            const Theta_t &Theta;
            const W_t &W;
            const X_t &X;

            dV_t &dV;
            V_t &V;
            b_t &b;

          public:

            constexpr Compute_Forward(const W_t &W, const Theta_t &Theta, const X_t &X, V_t &V, dV_t &dV, b_t &b)
                    : W(W), Theta(Theta), X(X), V(V), dV(dV), b(b) {

            }

            template<class idx_t>
            constexpr inline void compute_local_fields(const idx_t &idx, const size_t iter) {
                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            constexpr inline void compute_local_fields() {

            }

            template<typename T>
            constexpr inline void compute_activations() {
            }

        };

        template<class DeltaV_t, class DeltaX_t>
        class Compute_Backward {

            DeltaX_t &DeltaX;
            DeltaV_t &DeltaV;

          public:

            Compute_Backward(DeltaV_t &DeltaV, DeltaX_t &DeltaX) :
                    DeltaX(DeltaX), DeltaV(DeltaV) {

            }

            template<class Z_t, class V_t, class dV_t, class idx_t>
            constexpr inline void
            compute_deltas(const Z_t &Z, const V_t &V, const dV_t &dV, const idx_t &idx, const size_t iter) {

                const size_t mB = V.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;


            }

            template<class W_t, class dX_t>
            constexpr inline void compute_deltas(const W_t &W, const dX_t &dX) {

            }

        };

        template<class X_t, class Delta_t, class W_t, class dW_t, class Theta_t, class dTheta_t>
        class Compute_Update {

            W_t &W;
            dW_t &dW;
            Theta_t &Theta;
            dTheta_t &dTheta;

            const X_t &X;
            const Delta_t &Delta;

          public:

            Compute_Update(const X_t &X, const Delta_t &Delta, W_t &W, dW_t &dW, Theta_t &Theta, dTheta_t &dTheta) :
                    X(X), Delta(Delta), W(W), dW(dW), Theta(Theta), dTheta(dTheta) {

            }

            template<class idx_t>
            constexpr inline void compute_weights_update(const idx_t &idx, const size_t iter) {
                const size_t mB = Delta.size();
                const size_t x_begin = iter * mB;
                const size_t x_end = (iter + 1) * mB;

            }

            constexpr inline void compute_weights_update() {

            }

            constexpr inline void compute_bias_update() {

            }

            template<typename T>
            constexpr inline void apply_update(const T eta) {

            }

        };

    };

    template<typename T, mode_e MODE_E>
    class Engine {

      private:

        T eta;

      public:

        explicit Engine(T eta) : eta(eta) {};

        constexpr T get_eta() const { return eta; }

        template<debug_e DEBUG_E, class Engine_t, class Prev_Ptr_t, class Next_Ptr_t, class Network_Ptr_t>
        class ParametersEngine;

    };

    template<typename T, mode_e MODE_E>
    template<debug_e DEBUG_E, class Engine_t, class Prev_Ptr_t, class Next_Ptr_t, class Network_Ptr_t>
    class Engine<T, MODE_E>::ParametersEngine {
        static constexpr bool IS_DUMMY = MODE_E == DUMMY_E;
        static constexpr bool IS_SEQUENTIAL = MODE_E == SEQUENTIAL_E;
        static constexpr bool IS_MTHREADED = MODE_E == MTHREADED_E;
        static constexpr bool IS_MPROCESSING = MODE_E == MPROCESSING_E;


        using compute_t =
        std::conditional_t<IS_DUMMY, Compute_Dummy,
                std::conditional_t<IS_SEQUENTIAL, Compute_Sequential,
                        std::conditional_t<IS_MTHREADED, Compute_Mthreaded,
                                std::conditional_t<IS_MPROCESSING, Compute_Mprocessing, Compute_Mthreaded_Mprocessing>>>>;

        using Prev_t = std::remove_pointer_t<Prev_Ptr_t>;
        using Next_t = std::remove_pointer_t<Next_Ptr_t>;

        static constexpr NNet::layers_e PREV_EID = Prev_t::EID;
        static constexpr NNet::layers_e NEXT_EID = Next_t::EID;
        static constexpr size_t N = Next_t::N;
        static constexpr size_t M = Next_t::M;
        static constexpr size_t B = Prev_t::B;
        static constexpr size_t mB = Prev_t::mB;

        using T_prev = typename Prev_t::T;
        using T_next = typename Next_t::T;

        const Prev_Ptr_t prev;
        const Next_Ptr_t next;

        const Network_Ptr_t networkPtr;

        const Engine_t &engine;

      public:
        ParametersEngine(const Engine_t &engine, const Prev_Ptr_t prev, const Next_Ptr_t next, const Network_Ptr_t networkPtr);

         inline void compute_forward(size_t iter);

         inline void compute_backward(size_t iter);

         inline void compute_update(size_t iter);
    };

}

#include "CEng.hxx"

#endif //CPP_EXAMPLE_CENG_HPP
