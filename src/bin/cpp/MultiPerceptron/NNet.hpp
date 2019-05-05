//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_NNET_HPP
#define CPP_EXAMPLE_NNET_HPP


namespace NNet {

    enum layers_e {
        LAYER_BASE,
        LAYER_INPUT,
        LAYER_HIDDEN,
        LAYER_OUTPUT,
        LAYER_NULL,
        LAYERS,
        NODE
    };

    class LayerBase {


    public:
        static constexpr const char *NAME = "<LAYER BASE>";
        static constexpr layers_e EID = LAYER_BASE;
        static constexpr bool HAS_NEXT = false;


        std::string m_name = "<null>";

        const std::string m_type = "<null>";

        const std::pair<size_t, size_t> m_shape = {0, 0};

        virtual void print();

        template<typename T>
        LayerBase(T l, size_t N, size_t M, const char *_type);

        void make_name(size_t l);

        auto get_name();

        void print_name();

        auto alloc();

    };


    class LayerNull : public LayerBase {


    public:
        static constexpr const char *NAME = "<LAYER NULL>";
        static constexpr layers_e EID = LAYER_NULL;
        static constexpr bool HAS_NEXT = false;

        explicit LayerNull(size_t id);

        auto get_next();

    };


    template<typename T, size_t mB, size_t N, size_t M = 0, size_t ...>
    class LayerHidden : public LayerBase {


    public:
        static constexpr const char *NAME = "Layer Hidden";
        static constexpr layers_e EID = LAYER_HIDDEN;
        static constexpr bool HAS_NEXT = true;

        static constexpr size_t m_mB = mB;
        static constexpr size_t m_N = N;
        static constexpr size_t m_M = M;

        using m_T = T;

    public:

        Eigen::Matrix<T, M, N> W;
        Eigen::Matrix<T, M, N> dW;
        Eigen::Matrix<T, M, 1> Theta;
        Eigen::Matrix<T, M, 1> dTheta;
        std::array<Eigen::Matrix<T, M, 1>, mB> b;
        std::array<Eigen::Matrix<T, M, 1>, mB> V;
        std::array<Eigen::Matrix<T, M, 1>, mB> dV;
        std::array<Eigen::Matrix<T, M, 1>, mB> Delta;

    public:

        void print();

        LayerHidden(size_t id, const char *_type);

        explicit LayerHidden(size_t id);

        void print_W();

        void print_b();

        auto &get_W() { return W; };

        auto &get_dW() { return dW; };

        auto &get_Theta() { return Theta; };

        auto &get_dTheta() { return dTheta; };

        auto &get_V() { return V; };

        auto &get_dV() { return dV; };

        auto &get_b() { return b; };

        auto &get_Delta() { return Delta; };

        const auto &get_W() const { return W; };

        const auto &get_dW() const { return dW; };

        const auto &get_Theta() const { return Theta; };

        const auto &get_dTheta() const { return dTheta; };

        const auto &get_V() const { return V; };

        const auto &get_dV() const { return dV; };

        const auto &get_b() const { return b; };

        const auto &get_Delta() const { return Delta; };

        auto W_begin();

        auto W_end();

        auto dW_begin();

        auto dW_end();

        auto Delta_begin();

        auto Delta_end();

        auto V_begin();

        auto V_end();

        auto dV_begin();

        auto dV_end();

        auto b_begin();

        auto b_end();

        auto Theta_begin();

        auto Theta_end();

        auto dTheta_begin();

        auto dTheta_end();

    };

    template<typename T, size_t B, size_t mB, size_t N, size_t M>
    class LayerOutput : public LayerHidden<T, mB, N, M> {

    public:
        static constexpr const char *type = "Layer Output";
        static constexpr layers_e EID = LAYER_OUTPUT;
        static constexpr bool HAS_NEXT = false;

        static constexpr size_t m_B = B;
        static constexpr size_t m_mB = mB;
        static constexpr size_t m_N = N;
        static constexpr size_t m_M = M;

        using m_T = T;

    public:

        std::array<Eigen::Matrix<T, M, 1>, B> Z;

    public:

        auto &get_Z() { return Z; };

        const auto &get_Z() const { return Z; };

        auto Z_begin();

        auto Z_end();

        explicit LayerOutput(size_t id);

    };


    template<typename T, size_t B, size_t mB, size_t N>
    class LayerInput : public LayerBase {


    public:
        static constexpr const char *NAME = "Layer Input";
        static constexpr layers_e EID = LAYER_INPUT;

        static constexpr bool HAS_NEXT = true;
        static constexpr size_t m_mB = mB;
        static constexpr size_t m_B = B;
        static constexpr size_t m_N = N;

        using m_T = T;

        using v_t = typename Eigen::Matrix<T, N, 1>;

    public:

        std::vector<size_t> IDX;
        std::vector<v_t> V;

    public:

        void print();

        const auto get_next_mb() const;

        void shuffle_idx();

        const void shuffle_idx() const;

        auto &get_V() { return V; };

        const auto &get_idx() const { return IDX; };

        const auto &get_V() const { return V; };

        auto V_begin();

        auto V_end();

        explicit LayerInput(size_t id);

    };


    template<class Curr_t, class Next_t>
    class Node {


    public:
        static constexpr const char *NAME = "Node <" + Curr_t::NAME + " | " + Next_t::NAME + ">";
        static constexpr bool HAS_NEXT = Curr_t::HAS_NEXT;
        static constexpr layers_e EID = Curr_t::EID;


        static constexpr size_t N = Curr_t::m_N;
        static constexpr size_t M = Curr_t::m_M;
        static constexpr size_t mB = Curr_t::m_mB;
        static constexpr size_t B = Curr_t::m_B;

        using T = typename Curr_t::m_T;

        using m_Curr_t = Curr_t;
        using m_Next_t = Next_t;


    public:

        typedef Next_t *Next_Ptr_t;
        typedef Curr_t *Curr_Ptr_t;

        Curr_Ptr_t m_curr;
        Next_Ptr_t m_next;

        const size_t l;

    public:


        Node(size_t l, Curr_Ptr_t curr, Next_Ptr_t next);

        Node(size_t l, Next_Ptr_t next);

        void print();

        void print_name();

        auto get_next();


        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        void shuffle_idx() { m_curr->shuffle_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        const void shuffle_idx() const { m_curr->shuffle_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        auto &get_idx() { return m_curr->get_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        const auto &get_idx() const { return m_curr->get_idx(); };


        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        const auto get_next_mb() const { return m_curr->get_next_mb(); };

        auto &get_W() { return m_curr->get_W(); };

        auto &get_dW() { return m_curr->get_dW(); };

        auto &get_V() { return m_curr->get_V(); };

        auto &get_dV() { return m_curr->get_dV(); }

        auto &get_Z() { return m_curr->get_Z(); };

        auto &get_b() { return m_curr->get_b(); };

        auto &get_Theta() { return m_curr->get_Theta(); }

        auto &get_dTheta() { return m_curr->get_dTheta(); }

        auto &get_Delta() { return m_curr->get_Delta(); }

        const auto &get_W() const { return m_curr->get_W(); };

        const auto &get_dW() const { return m_curr->get_dW(); };

        const auto &get_V() const { return m_curr->get_V(); };

        const auto &get_dV() const { return m_curr->get_dV(); }

        const auto &get_Z() const { return m_curr->get_Z(); };

        const auto &get_b() const { return m_curr->get_b(); };

        const auto &get_Theta() const { return m_curr->get_Theta(); }

        const auto &get_dTheta() const { return m_curr->get_dTheta(); }

        const auto &get_Delta() const { return m_curr->get_Delta(); }


    };


    template<class Curr_t, class Next_Linked_Ptr_t>
    class Linked {


    public:

        static constexpr const char *NAME = "Linked <" + Curr_t::NAME + " --> " + Next_Linked_Ptr_t::NAME + ">";
        static constexpr bool HAS_NEXT = Curr_t::HAS_NEXT;
        static constexpr layers_e EID = Curr_t::EID;

        static constexpr size_t N = Curr_t::m_N;
        static constexpr size_t M = Curr_t::m_M;
        static constexpr size_t mB = Curr_t::m_mB;
        static constexpr size_t B = Curr_t::m_B;

        using T = typename Curr_t::m_T;

        using m_Curr_t = Curr_t;
        using m_Next_Linked_Ptr_t = Next_Linked_Ptr_t;

        using Next_Linked_t  =  std::remove_pointer_t<Next_Linked_Ptr_t>;

        using Curr_Ptr_t =  std::add_pointer_t<Curr_t>;

        using node_t = Node<Curr_t, Next_Linked_t>;

        using node_ptr_t =  std::add_pointer_t<node_t>;

    private:

        node_ptr_t m_head;
        const size_t l;

    public:

        Linked(Curr_Ptr_t curr, Next_Linked_Ptr_t next, size_t l);

        Linked(Next_Linked_Ptr_t next, size_t l);

        constexpr auto has_next() {

            return HAS_NEXT;
        }

        constexpr auto eid() {

            return eid;
        }

        void print();

        void print_name();

        auto get_name();

        auto get_curr();

        auto get_next();

        void print_V();

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        void shuffle_idx() { m_head->shuffle_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        const void shuffle_idx() const { m_head->shuffle_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        auto &get_idx() { return m_head->get_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        const auto & get_idx() const { return m_head->get_idx(); };

        template<typename = std::enable_if_t<EID == LAYER_INPUT>>
        const auto get_next_mb() const { return m_head->get_next_mb(); };

        auto &get_W() { return m_head->get_W(); };

        auto &get_dW() { return m_head->get_dW(); };

        auto &get_V() { return m_head->get_V(); };

        auto &get_dV() { return m_head->get_dV(); }

        auto &get_Z() { return m_head->get_Z(); };

        auto &get_b() { return m_head->get_b(); };

        auto &get_Theta() { return m_head->get_Theta(); }

        auto &get_dTheta() { return m_head->get_dTheta(); }

        auto &get_Delta() { return m_head->get_Delta(); }

        const auto &get_W() const { return m_head->get_W(); };

        const auto &get_dW() const { return m_head->get_dW(); };

        const auto &get_V() const { return m_head->get_V(); };

        const auto &get_dV() const { return m_head->get_dV(); }

        const auto &get_Z() const { return m_head->get_Z(); };

        const auto &get_b() const { return m_head->get_b(); };

        const auto &get_Theta() const { return m_head->get_Theta(); }

        const auto &get_dTheta() const { return m_head->get_dTheta(); }

        const auto &get_Delta() const { return m_head->get_Delta(); }

        auto W_begin();

        auto W_end();

        auto V_begin();

        auto V_end();

        auto Z_begin();

        auto Z_end();

        auto dV_begin();

        auto dV_end();

        auto b_begin();

        auto b_end();

        auto dW_begin();

        auto dW_end();

        auto Delta_begin();

        auto Delta_end();

        auto Theta_begin();

        auto Theta_end();

        auto dTheta_begin();

        auto dTheta_end();

    };


    class LayersBase {

    public:

        size_t nFullBatch = 0;

        size_t nMiniBatch = 0;

        size_t nLayers = 0;

        size_t nHidden = 0;

        virtual void print();

        LayersBase();

    };


    template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
    class LayersMaker : public LayersBase {


    public:

        using m_T_IN = T_IN;
        using m_T = T;

        static constexpr size_t m_B = B;

        static constexpr size_t m_mB = mB;

        static constexpr size_t m_I = I;

        static constexpr size_t nHidden = sizeof ... (shapeNN);
        static constexpr size_t nLayers = nHidden + 1;
        static constexpr size_t m_shapeNN[nHidden] = {shapeNN...};

    public:
        template<class Layers_Ptr>
        auto print(Layers_Ptr layersPtr);

        template<debug_e DEBUG_E, size_t N>
        auto alloc(size_t l);

        template<debug_e DEBUG_E, size_t N, size_t M>
        auto alloc(size_t l);

        template<debug_e DEBUG_E, size_t N, size_t M, size_t K, size_t ... nextShapeNN>
        auto alloc(size_t l);

        template<debug_e DEBUG_E = DEBUG1>
        auto alloc();

        LayersMaker();


    };

    enum compute_e {
        FORWARD_E,
        BACKWARD_E,
        UPDATE_E
    };

    template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
    class Network : public Layers_t {

    public:
        Layers_t layers;
        Linked_Ptr_t linkedPtr;

        Engine_Ptr enginePtr;
        Datain_Ptr datainPtr;
        Datagen_Ptr datagenPtr;


    public:


        static constexpr size_t mB = Layers_t::m_mB;
        static constexpr size_t B = Layers_t::m_B;
        static constexpr size_t ITER = B / mB;


        using T_IN = typename Layers_t::m_T_IN;
        using T = typename Layers_t::m_T;

    public:
        Network(Layers_t &layers, Linked_Ptr_t &linkedPtr, Engine_Ptr &enginePtr, Datain_Ptr &datainPtr,
                Datagen_Ptr &datagenPtr);

        template<debug_e DEBUG_E, compute_e COMPUTE_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
        void compute(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l);

        template<debug_e, compute_e COMPUTE_E, class Curr_Linked_Ptr_t>
        void compute(Curr_Linked_Ptr_t curr, size_t l, size_t epoch, size_t iter);

        template<debug_e DEBUG_E>
        void compute(size_t epochs);

        template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
        auto init_parameters(Curr_Linked_Ptr_t linkedPtr);

        template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
        auto init_patterns(Curr_Linked_Ptr_t linkedPtr);

        template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
        auto init_classifiers(Curr_Linked_Ptr_t layersPtr);

        template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
        auto init(Curr_Linked_Ptr_t curr, size_t l);

        template<debug_e DEBUG_E>
        auto init();

        void print();
    };

}

#include "NNet.hxx"

#endif //CPP_EXAMPLE_NNET_HPP
