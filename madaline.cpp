#include "madaline.hpp"

using namespace std;

Madaline::Madaline(vector<unsigned> &topology) {
	for(unsigned l = 0; l < topology.size(); ++l) {
		c_layers.push_back(Layer());
		unsigned layer_size = topology[l];
		for(unsigned n = 0; n < layer_size; ++n) {
			// first layer is input layer
			unsigned n_of_inputs = (l == 0) ? 0 : topology[l - 1];
			c_layers[l].push_back(Adaline(n, n_of_inputs));
		}
	}
}

void Madaline::forward(const vector<double> &input) {
	assert(input.size() == c_layers[0].size());
	
	// first layer
	// set output of neurons to input values
	for(unsigned n = 0; n < input.size(); ++n) {
		c_layers[0][n].set_output(input[n]);
	}
	
	// all other layers
	for(unsigned l = 1; l < c_layers.size(); ++l) {
		Layer &this_layer = c_layers[l];
		Layer &previous_layer = c_layers[l - 1];
		
		// propagate for each neuron in this layer
		for(unsigned n = 0; n < this_layer.size(); ++n) {
			this_layer[n].forward(previous_layer);
		}
	}
}

void Madaline::get_result(vector<double> &results) {
	results.clear();
	Layer &output_layer = c_layers.back();
	for(unsigned n = 0; n < output_layer.size(); ++n) {
		double neuron_output = output_layer[n].get_output();
		results.push_back(neuron_output);
	}
}

void Madaline::back_propagate(const vector<double> &target_results) {
	Layer &output_layer = c_layers.back();
	assert(output_layer.size() == target_results.size());

	c_net_error.clear();
	for(unsigned n = 0; n < output_layer.size(); ++n) {
		c_net_error.push_back(target_results[n] - output_layer[n].get_output());
	}
	// TODO: pokracovat
}

Madaline::~Madaline() {}

/***************************  debug functions  *************************/
void Madaline::print_network() {
	for(unsigned l = 0; l < c_layers.size(); ++l) {
		cout << "Layer " << l << ":" << endl;
		for(unsigned n = 0; n < c_layers[l].size(); ++n) {
			cout << "\t";
			c_layers[l][n].print_neuron();
		}
	}
}
