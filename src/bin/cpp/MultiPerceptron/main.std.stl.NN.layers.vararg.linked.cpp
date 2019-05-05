//
// Created by a on 3/22/19.
//

#include <vector>
#include <iostream>
#include <map>
#include <cstring>
#include <list>
#include <memory>
#include <chrono>
#include <thread>
#include <cassert>
#include <csignal>
#include <cmath>
#include <variant>

#define DEFN 10000


class LayerBase {


  public:

	std::string type;

	std::string name;

	std::pair<size_t, size_t> shape;

	virtual void print() {

		std::cout << name << std::endl;
	}

	LayerBase(size_t l, size_t N, size_t M) : shape {N, M} {

		if (N == 0) {
			type = "Input Layer";
		} else if (M == 0) {
			type = "Output Layer";
		} else {
			type = "Hidden Layer";
		}

		make_name(l);


		std::cout << "LayerBase : name = "
		          << name << std::endl;
	}

	void make_name(size_t l) {

		name = "[" + type + " : " + std::to_string(l) + "]:(" + std::to_string(shape.first) + "," +
		       std::to_string(shape.second) + ")";
	}

};


#define sint size_t

template<class Base, class Derived>
class CRTP : public Base {

};

// This CRTP class implements clone() for Derived
template<class Derived, size_t ... shapeNN>
class LayerBaseCRTP : public LayerBase {

  public:


	virtual Derived *caster() const {

		std::cout << "Cloning!" << std::endl;
		//return static_cast<Derived const &>(*this);
	}


	//LayerBase_CRTP(size_t l, size_t N, size_t M) : LayerBase(l, N, M) {}

	void interface() {

		static_cast<Derived *>(this)->implementation();
	}

	void implementation() {

		std::cout << "Implementation Base" << std::endl;
	}

};


template<class Prev, typename T, size_t N, size_t ...>
class LayerOutput : public LayerBase {
  public:

	std::array<T, N * 1> V;

	std::shared_ptr<Prev> LP;

  public:



	LayerOutput(size_t id) : LayerBase(id, N, 0) {


		std::cout << "make output : " << id << ", : N = " << N << std::endl;

	}

	void set_prev(Prev * L){
		LP = static_cast<decltype(LP)>(L);
		assert(LP != nullptr);

	}


};

const bool is_greater(const int a, const int b) {

	return a > b;
}

template<typename T, size_t mB, size_t N, size_t M = 0, size_t ... shapeNN>
class LayerHidden : public LayerBase {
  public:


	//map_type x;

	typedef LayerHidden<T, mB, N, M, shapeNN...> this_t;

	typedef LayerHidden<T, mB, M, shapeNN...> LH_t;

	typedef LayerOutput<T, this_t, M> LO_t;

	std::unique_ptr<LH_t> LH;

	std::unique_ptr<LO_t> LO;

	static const int lenShapeNN = sizeof ... (shapeNN);

	static const bool hasMoreLayers = lenShapeNN > 0;

	typedef typename std::conditional<sizeof ... (shapeNN) != 0, LH_t, LO_t>::type LN_t;

	typedef std::variant<LH_t, LO_t> LN_vt;


	std::unique_ptr<LN_t> LN;

	std::array<T, M * N> W;

	std::array<T, M * N> dW;

	std::array<T, M * mB> V;

	std::array<T, M * mB> dV;

	std::array<T, M * mB> b;

	std::array<T, M * mB> Delta;

	std::array<T, M * 1> Theta;

	std::array<T, M * 1> dTheta;

  public:

	void print() {

		LayerBase::print();

		LN->print();

	}

	LayerHidden(size_t id, LayerBase *L) : LayerBase(id, N, M), LN(static_cast<LN_t *> (L)) {

		assert(LN != nullptr);

	}
};


template<typename T, size_t mB, size_t N, size_t ...shapeNN>
class LayerInput : public LayerBase {


  public:
	std::array<T, N * mB> V;

	typedef LayerHidden<T, mB, N, shapeNN...> LH_t;

	std::unique_ptr<LH_t> LH;

  public:

	void print() {

		LayerBase::print();
		LH->print();
	}

	LayerInput(size_t id, LayerBase *LH) : LH(static_cast<LH_t *>(LH)), LayerBase(id, 0, N) {

		assert(LH != nullptr);
		std::cout << "make input : " << id << ", : N = " << N << std::endl;
	}

};

template <class Curr, class Next>
class LayerLinkedNext : Curr {




};

template <class Prev, class Curr>
class LayerLinkedPrev : Curr {



};

template<class Prev, class Curr, class Next>
class LayerDoubleLinked : Curr {

	std::shared_ptr<Prev> LP;
	std::shared_ptr<Next> LN;


};

class LayersBase {

  public:

	sint nFullBatch = 0;

	sint nMiniBatch = 0;

	sint nLayers = 0;

	sint nHidden = 0;

	virtual void print() {

		std::cout << "in base" << std::endl;
	}

};

template<typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
class Layers : public LayersBase {

	typedef LayerInput<T, mB, I, shapeNN...> LI_t;

	std::unique_ptr<LI_t> layerInput;

	typedef typename std::conditional<
	        sizeof ... (shapeNN) != 0,
	        LayerHidden<T, mB, I, shapeNN...>,
	        LayerOutput<LI_t, T, mB, I>
	        >::type LN_vt;

  public:


	template<class LP_t, size_t M >
	auto make_next_layer(size_t l) {

		std::cout << "last layer : " << l << "/" << nLayers << " : " << " M = " << M << std::endl;

		typedef LayerOutput<T, M> LO_t;

		return new LO_t(l);

	}


	template<class LP_t, size_t N, size_t M, size_t ... nextShapeNN>
	auto make_next_layer(size_t l) {

		typedef LayerHidden<T, mB, N, M, nextShapeNN...> cLH_t;
		typedef LayerHidden<T, mB, M, nextShapeNN...> nLH_t;
		typedef LayerOutput<T, M> nLO_t;

		typedef typename std::conditional<sizeof ... (nextShapeNN) != 0, nLH_t, nLO_t>::type nL_t;

		//typedef LayerHiddenCRTP<cLH_t, nL_t> nLH_crtp;

		auto NL = make_next_layer<cLH_t, M, nextShapeNN...>(l + 1);

		return new cLH_t(l, NL);
	}


	Layers() {

		std::cout << " LayersMaker... " << std::endl;
		this->nHidden = sizeof ... (shapeNN);
		this->nLayers = this->nHidden + 1;

		this->nMiniBatch = mB;
		this->nFullBatch = B;
		auto NL = make_next_layer<LI_t, I, shapeNN...>(1);


		layerInput = std::make_unique<LI_t>(0, std::move(NL));

	}

	void print() {

		layerInput->print();
	}

};


int main() {

	const size_t B = 100;
	const size_t mB = 10;

	LayersBase *layers = new Layers<int, B, mB, 1000, 100, 100, 30, 10>();

	//auto * lh = (LayerHidden<int, 100, 100, mB> * ) layersWrap.layersTraining[1];
	//lh->V;
	layers->print();


	std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	return 0;
}
