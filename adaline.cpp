#include "adaline.hpp"

using namespace std;

Adaline::Adaline(const unsigned index, const unsigned n_of_inputs) {
	initialize_weights(n_of_inputs);
	c_index = index;
}

void Adaline::initialize_weights(unsigned n_of_inputs) {
	for(unsigned i = 0; i < n_of_inputs; ++i) {
		c_weights.push_back(rand() / double(RAND_MAX));
	}
	// bias
	c_weights.push_back(1.0);
}

void Adaline::forward(const Layer &input_layer) {
	assert(input_layer.size() == c_weights.size() - 1);
	double sum = 0.0;
	
	// sum the input_layer's outputs (times weights)
	for (unsigned n = 0; n < input_layer.size(); ++n) {
		sum += input_layer[n].get_output() * c_weights[n];
	}
	// include bias
	sum += c_weights.back();

	c_output = activation_function(sum);
}

double Adaline::get_output() const {
	return c_output;
}
void Adaline::set_output(const double val) {
	c_output = val;
}

double Adaline::activation_function(const double val) {
	return 1.0 / (1.0 + exp(-val));
}

Adaline::~Adaline() {}

/***************************  debug functions  *************************/
void Adaline::print_neuron() {
	cout << "Neuron " << c_index << ": ";
	for(unsigned i = 0; i < c_weights.size(); ++i) {
		cout << c_weights[i] << "  ";
	}
	cout << endl;
}
