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
		std::array<Eigen::Matrix<T, M, 1>, mB> V;
		std::array<Eigen::Matrix<T, M, 1>, mB> dV;
		std::array<Eigen::Matrix<T, M, 1>, mB> b;
		std::array<Eigen::Matrix<T, M, 1>, mB> Delta;

	  public:

		void print();

		LayerHidden(size_t id, const char *_type);

		explicit LayerHidden(size_t id);

		void print_W();

		void print_b();

		const auto &get_W() const { return W; };

		auto &get_W() { return W; }

		const auto &get_V() const { return V; };

		auto &get_V() { return V; };

		const auto &get_dV() const { return dV; };

		auto &get_dV() { return dV; };

		const auto &get_dW() const { return dW; };

		auto &get_dW() { return dW; };

		const auto &get_Delta() const { return Delta; };

		auto &get_Delta() { return Delta; };

		const auto &get_b() const { return b; };

		auto &get_b() { return b; };

		const auto &get_Theta() const { return Theta; };

		auto &get_Theta() { return Theta; };

		const auto &b_begin() const { return V.cdata(); };

		auto b_begin() { return V.data(); };

		const auto &b_end() const { return V.cdata() + V.size(); };

		auto b_end() { return V.data() + V.size(); };

		const auto &V_begin() const { return V.cdata(); };

		auto V_begin() { return V.data(); };

		const auto &V_end() const { return V.cdata() + V.size(); };

		auto V_end() { return V.data() + V.size(); };

		const auto &dV_begin() const { return V.cdata(); };

		auto dV_begin() { return V.data(); };

		const auto &dV_end() const { return V.cdata() + V.size(); };

		auto dV_end() { return V.data() + V.size(); };

		const auto &W_begin() const { return W.cdata(); };

		auto W_begin() { return W.data(); };

		const auto &W_end() const { return W.cdata() + W.size(); };

		auto W_end() { return W.data() + W.size(); };

		const auto &dW_begin() const { return dW.cdata(); };

		auto dW_begin() { return dW.data(); };

		const auto &dW_end() const { return dW.cdata() + dW.size(); };

		auto dW_end() { return dW.data() + dW.size(); };

		const auto &Theta_begin() const { return Theta.cdata(); };

		auto Theta_begin() { return Theta.data(); };

		const auto &Theta_end() const { return Theta.cdata() + Theta.size(); };

		auto Theta_end() { return Theta.data() + Theta.size(); };

		const auto &dTheta_begin() const { return dTheta.cdata(); };

		auto dTheta_begin() { return dTheta.data(); };

		const auto &dTheta_end() const { return dTheta.cdata() + dTheta.size(); };

		auto dTheta_end() { return dTheta.data() + dTheta.size(); };

		const auto &Delta_begin() const { return Delta.cdata(); };

		auto Delta_begin() { return Delta.data(); };

		const auto &Delta_end() const { return Delta.cdata() + Delta.size(); };

		auto Delta_end() { return Delta.data() + Delta.size(); };
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
		const auto &get_Z() const { return Z; };

		auto &get_Z() { return Z; };

		const auto &Z_begin() const { return Z.cdata(); };

		auto Z_begin() { return Z.data(); };

		const auto &Z_end() const { return Z.cdata() + Z.size(); };

		auto Z_end() { return Z.data() + Z.size(); };

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

	  public:

		std::array<Eigen::Matrix<T, N, 1>, B> V;

	  public:

		void print();

		const auto &get_V() const { return V; };

		auto &get_V() { return V; };

		const auto &V_begin() const { return V.cdata(); };

		auto V_begin() { return V.data(); };

		const auto &V_end() const { return V.cdata() + V.size(); };

		auto V_end() { return V.data() + V.size(); };

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

		using m_Curr_t = Curr_t;
		using m_Next_t = Next_t;

		using m_T = typename Curr_t::m_T;

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

		using m_T = typename Curr_t::m_T;

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

		const auto &get_W() const { return m_head->m_curr->get_W(); };

		auto &get_W() { return m_head->m_curr->get_W(); };

		const auto &get_dW() const { return m_head->m_curr->get_dW(); };

		auto &get_dW() { return m_head->m_curr->get_dW(); };

		const auto &get_V() const { return m_head->m_curr->get_V(); };

		auto &get_V() { return m_head->m_curr->get_V(); };

		const auto &get_dV() const { return m_head->m_curr->get_dV(); };

		auto &get_dV() { return m_head->m_curr->get_dV(); };

		const auto &get_Z() const { return m_head->m_curr->get_Z(); };

		auto &get_Z() { return m_head->m_curr->get_Z(); };

		const auto &get_b() const { return m_head->m_curr->get_b(); };

		auto &get_b() { return m_head->m_curr->get_b(); };

		const auto &get_Delta() const { return m_head->m_curr->get_Delta(); };

		auto &get_Delta() { return m_head->m_curr->get_Delta(); };

		const auto &get_Theta() const { return m_head->m_curr->get_Theta(); };

		auto &get_Theta() { return m_head->m_curr->get_Theta(); };

		const auto &get_dTheta() const { return m_head->m_curr->get_dTheta(); };

		auto &get_dTheta() { return m_head->m_curr->get_dTheta(); };

		const auto &W_begin() const { return m_head->m_curr->W_begin(); };

		auto W_begin() { return m_head->m_curr->W_begin(); };

		const auto &W_end() const { return m_head->m_curr->W_end(); };

		auto W_end() { return m_head->m_curr->W_end(); };

		const auto &V_begin() const { return m_head->m_curr->V_begin(); };

		auto V_begin() { return m_head->m_curr->V_begin(); };

		const auto &V_end() const { return m_head->m_curr->V_end(); };

		auto V_end() { return m_head->m_curr->V_end(); };

		const auto &dV_begin() const { return m_head->m_curr->dV_begin(); };

		auto dV_begin() { return m_head->m_curr->dV_begin(); };

		const auto &dV_end() const { return m_head->m_curr->dV_end(); };

		auto dV_end() { return m_head->m_curr->dV_end(); };

		const auto &Z_begin() const { return m_head->m_curr->Z_begin(); };

		auto Z_begin() { return m_head->m_curr->Z_begin(); };

		const auto &Z_end() const { return m_head->m_curr->Z_end(); };

		auto Z_end() { return m_head->m_curr->Z_end(); };

		const auto &b_begin() const { return m_head->m_curr->b_begin(); };

		auto b_begin() { return m_head->m_curr->b_begin(); };

		const auto &b_end() const { return m_head->m_curr->b_end(); };

		auto b_end() { return m_head->m_curr->b_end(); };

		const auto &Theta_begin() const { return m_head->m_curr->Theta_begin(); };

		auto Theta_begin() { return m_head->m_curr->Theta_begin(); };

		const auto &Theta_end() const { return m_head->m_curr->Theta_end(); };

		auto Theta_end() { return m_head->m_curr->Theta_end(); };

		const auto &dTheta_begin() const { return m_head->m_curr->dTheta_begin(); };

		auto dTheta_begin() { return m_head->m_curr->dTheta_begin(); };

		const auto &dTheta_end() const { return m_head->m_curr->dTheta_end(); };

		auto dTheta_end() { return m_head->m_curr->dTheta_end(); };

		const auto &Delta_begin() const { return m_head->m_curr->Delta_begin(); };

		auto Delta_begin() { return m_head->m_curr->Delta_begin(); };

		const auto &Delta_end() const { return m_head->m_curr->Delta_end(); };

		auto Delta_end() { return m_head->m_curr->Delta_end(); };

		void print_self_size() {

			std::cout << "[SIZE] Linked : " << sizeof(*this) << std::endl;
		}

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

		static constexpr size_t m_mB = mB;
		static constexpr size_t m_B = B;
		static constexpr size_t m_I = I;

		static constexpr size_t nHidden = sizeof ... (shapeNN);
		static constexpr size_t nLayers = nHidden + 1;
		static constexpr size_t m_shapeNN[nHidden] = {shapeNN...};

	  public:

		void print_self_size() {

			std::cout << "[SIZE] LayersMaker : " << sizeof(*this) << std::endl;
		}

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

	template<class Layers_Ptr_t, class Linked_Ptr_t, class Datain_Ptr_t, class Datagen_Ptr_t>
	class Network : public std::remove_pointer_t<Layers_Ptr_t> {

	  public:
		Layers_Ptr_t layersPtr;
		Linked_Ptr_t linkedPtr;

		Datain_Ptr_t datainPtr;
		Datagen_Ptr_t datagenPtr;

	  public:

		static constexpr size_t mB = Layers_Ptr_t::m_mB;
		static constexpr size_t B = Layers_Ptr_t::m_B;

		using Layers_t = std::remove_pointer_t<Layers_Ptr_t>;

		using T_IN = typename Layers_t::m_T_IN;
		using T = typename Layers_t::m_T;

	  public:
		Network(Layers_Ptr_t layersPtr, Linked_Ptr_t linkedPtr, Datain_Ptr_t datainPtr, Datagen_Ptr_t datagenPtr);

		template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
		auto forward(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l);

		template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
		auto backward(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l);

		template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
		auto deltas(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l);

		template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
		auto compute(Curr_Linked_Ptr_t curr, size_t l);

		template<debug_e DEBUG_E>
		auto compute();

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

		void print_self_size() {

			std::cout << "[SIZE] Network : " << sizeof(*this) << std::endl;
		}
	};

}

#include "NNet.hxx"

#endif //CPP_EXAMPLE_NNET_HPP
