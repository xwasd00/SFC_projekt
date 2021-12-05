#include "adaline.hpp"

using namespace std;

Adaline::Adaline(const unsigned index, const unsigned n_of_inputs, const double &mi, const double &eps) {
	this->c_mi = mi;
	this->c_epsilon = eps;
	this->initialize_weights(n_of_inputs);
	c_index = index;
}

Adaline::Adaline(const unsigned index, const unsigned n_of_inputs) {
	this->initialize_weights(n_of_inputs);
	c_index = index;
}

void Adaline::initialize_weights(unsigned n_of_inputs) {
	// set weights 
	for(unsigned i = 0; i < n_of_inputs; ++i) {
		c_weights.push_back(rand() / double(RAND_MAX));
		//c_weights.push_back(0);
	}
	// including bias
	c_weights.push_back(1);
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

	if(c_add_epsilon) {
		sum += c_epsilon;
	}

	c_potential = sum;
	c_output = activation_function(c_potential);
}

double Adaline::get_output() const {
	return c_output;
}
void Adaline::set_output(const double val) {
	c_output = val;
}

double Adaline::activation_function(const double val) {
	return (1 / (1 + exp(-val)));
	//return tanh(val);
}

void Adaline::add_epsilon() {
	c_add_epsilon = true;
}

void Adaline::remove_epsilon() {
	c_add_epsilon = false;
}

void Adaline::update_weights(const Layer &previous_layer, const double net_e, const double neuron_net_e) {
	double delta = - 2 * c_mi * net_e * ((neuron_net_e - net_e) / c_epsilon);
	
	for(unsigned n = 0; n < previous_layer.size(); ++n) {
		c_weights[n] += delta * previous_layer[n].get_output(); 
	}
	// update bias
	c_weights.back() += delta * 1;
}

void Adaline::save_weights(std::ofstream &file){
	file << c_index << " [ ";
	for(unsigned i = 0; i < c_weights.size(); ++i) {
		file << c_weights[i] << " ";
	}
	file << "] ";
}

bool Adaline::load_weights(std::stringstream &ss) {
	string s;
	unsigned index;
	if (ss >> index){
		ss >> s;
		c_weights.clear();
		double weight;
		while (ss >> weight) {
			c_weights.push_back(weight);
		}
		ss >> s;
		return true;
	}
	return false;
}



Adaline::~Adaline() {}

/***************************  debug functions  *************************/
void Adaline::print_neuron() {
	cout << "N" << c_index << "[ ";
	for(unsigned i = 0; i < c_weights.size(); ++i) {
		cout << c_weights[i] << " ";
	}
	cout << "]" << endl;
}
