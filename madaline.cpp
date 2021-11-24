#include "madaline.hpp"

using namespace std;

Madaline::Madaline(const vector<unsigned> &topology) {
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

void Madaline::train(const vector<t_Sample> &train_data) {
	double max_error = 0;
	unsigned iterations = 0;
	vector<double> results;
	double THRESHOLD = 0.1;

	do {
		max_error = 0;
		for (unsigned i = 0; i < train_data.size(); ++i) {
			this->forward(train_data[i].input);
			this->get_result(results);
			if (error(results, train_data[i].desired_output) > THRESHOLD) {
				this->update_network(train_data[i].desired_output);
				max_error = max(abs(c_net_error), max_error);
			}
		}
		//cout << "max error: " << max_error << " iterations: " << iterations << endl;
		++iterations;
	} while (max_error > THRESHOLD);
}

void Madaline::update_network(const vector<double> &desired_results) {
	vector<double> output;
	vector<double> altered_output;
	double neuron_net_error;

	this->get_result(output);
	assert(output.size() == desired_results.size());

	c_net_error = this->error(desired_results, output);

	// each layer, except 'input'
	for(unsigned l = 1; l < c_layers.size(); ++l) {

		// each neuron in layer 'l'
		for(unsigned n = 0; n < c_layers[l].size(); ++n) {
			Adaline &neuron = c_layers[l][n];
			Layer &previous_layer = c_layers[l - 1];
			neuron.add_epsilon();
			this->partial_forward(l, altered_output);
			neuron.remove_epsilon();
			neuron.forward(previous_layer); // restore output of neuron

			neuron_net_error = this->error(desired_results, altered_output);
			neuron.update_weights(previous_layer, c_net_error, neuron_net_error);
		}
	}
}

void Madaline::partial_forward(const unsigned layer_index, vector<double> &output) {
	for(unsigned l = layer_index; l < c_layers.size(); ++l) {
		Layer &this_layer = c_layers[l];
		Layer &previous_layer = c_layers[l - 1];
		
		// propagate for each neuron in this layer
		for(unsigned n = 0; n < this_layer.size(); ++n) {
			this_layer[n].forward(previous_layer);
		}
	}
	this->get_result(output);
}

double Madaline::error(const vector<double> &a, const vector<double> &b) {
	assert(a.size() == b.size());
	double sum = 0.0;
	for(unsigned i = 0; i < a.size(); ++i) {
		sum += abs(a[i] - b[i]);
	}
	return sum;
}

Madaline::~Madaline() {}

/***************************  debug functions  *************************/
void Madaline::print_network() {
	cout << "TOPOLOGY:" << endl;
	for(unsigned l = 0; l < c_layers.size(); ++l) {
		cout << "Layer " << l;
		if (l == 0) {
			cout << "(input layer - has only biases)";
		}
		cout << ":" << endl;
		for(unsigned n = 0; n < c_layers[l].size(); ++n) {
			cout << "\t";
			c_layers[l][n].print_neuron();
		}
	}
}

void Madaline::print_output() {
	Layer &output_layer = c_layers.back();
	cout << "OUTPUT: [ ";
	for(unsigned n = 0; n < output_layer.size(); ++n) {
		cout << output_layer[n].get_output() << " ";
	}
	cout << "]" << endl;
}

void Madaline::print_input() {
	Layer &input_layer = c_layers[0];
	cout << "INPUT: [ ";
	for(unsigned n = 0; n < input_layer.size(); ++n) {
		cout << input_layer[n].get_output() << " ";
	}
	cout << "]" << endl;
}

void Madaline::print_response_on_train_data(const vector<t_Sample> &train_data) {
	cout << "---------------------------------------------" << endl;
	cout << "TRAINING OUTPUT:" << endl;
	this->print_network();
	cout << endl;
	cout << "DATA:" << endl;
	vector<double> results;
	for (unsigned i = 0; i < train_data.size(); ++i) {
		this->forward(train_data[i].input);
		cout << "\t";
		this->print_input();
		cout << "\t";
		this->print_output();
		cout << "\tDESIRED: [ ";
		for(unsigned j = 0; j < train_data[i].desired_output.size(); ++j) {
			cout << train_data[i].desired_output[j] << " ";
		}
		cout << "]" << endl << endl;
	}
	cout << "---------------------------------------------" << endl;

}