//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_CENG_HPP
#define CPP_EXAMPLE_CENG_HPP

namespace CEng {

	template<typename T>
	class Activation {

	  public:

		virtual T g(T x) const = 0;
		virtual T dg(T x) const = 0;

		void print_name() {
			std::cout << "Activation" << std::endl;
		}
	};

	template<typename T>
	class FastSigmoid : Activation<T> {

	  public:

		constexpr T g(const T x) const override {
			return x / (T{1} + std::abs(x));
		}

		constexpr T dg(const T dx) const override {
			return dx * (T{1} - dx);
		}

		void print_name() {
			std::cout << "FastSigmoid" << std::endl;
		}

	};

	template<typename T>
	class Sigmoid : Activation<T> {

	  public:

		constexpr T g(const T x) const override {
			return T{1} / (T{1} + std::exp(-x));
		}

		constexpr T dg(const T dx) const override {
			return dx * (T{1} - dx);
		}

		void print_name() {
			std::cout << "Sigmoid" << std::endl;
		}

	};

	template<typename T>
	class Tanh : Activation<T> {

	  public:

		constexpr T g(const T x) const override {
			return std::tanh(x);
		}

		constexpr T dg(const T dx) const override {
			return 1 - std::pow(dx, 2);
		}

		void print_name() {
			std::cout << "Tanh" << std::endl;
		}
	};

	template<typename T>
	class Relu : Activation<T> {

	  public:

		constexpr T g(const T x) const override {
			return std::max(0, x);
		}

		constexpr T dg(const T dx) const override {
			return dx > 0;
		}

		void print_name() {
			std::cout << "Relu" << std::endl;
		}
	};

	enum activation_e {
		TANH_E,
		SIGMOID_E,
		FASTSIGMOID_E,
		RELU_E,
	};

	template<debug_e DEBUG_E, activation_e ACTIVATION_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	class Engine {

	  private:

		using Prev_Linked_t = std::remove_pointer_t<Prev_Linked_Ptr_t>;
		using Next_Linked_t = std::remove_pointer_t<Next_Linked_Ptr_t>;

		using T_prev = typename Prev_Linked_t::m_T;
		using T_next = typename Next_Linked_t::m_T;

		static constexpr size_t mB = Prev_Linked_t::mB;
		static constexpr size_t N = Next_Linked_t::N;
		static constexpr size_t M = Next_Linked_t::M;
		static constexpr NNet::layers_e NEXT_EID = Next_Linked_t::EID;
		static constexpr NNet::layers_e PREV_EID = Prev_Linked_t::EID;
		static constexpr bool isOutputLayer = NEXT_EID == NNet::LAYER_OUTPUT;
		static constexpr bool isInputLayer = PREV_EID == NNet::LAYER_INPUT;

		Prev_Linked_Ptr_t prevLinkedPtr;
		Next_Linked_Ptr_t nextLinkedPtr;

	  public:
		void print_self_size() {
			std::cout << "[SIZE] Engine : " << sizeof(*this) << std::endl;
		}

		Engine(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr);

		auto compute_forward();
		auto compute_backward();

	};

}

#include "CEng.hxx"

#endif //CPP_EXAMPLE_CENG_HPP
