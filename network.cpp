#include "network.hpp"


/********************** Connection ***********************/
Connection::Connection() {
	weight = random_weight();
}
double Connection::random_weight() {
	return rand() / double(RAND_MAX);
}

/************************ Neuron *************************/
Neuron::Neuron(unsigned n_of_outputs, unsigned index) {
	for (unsigned c = 0; c < n_of_outputs; ++c) {
		output_weights.push_back(Connection());
	}
	my_index = index;
}
void Neuron::Forward(const Layer &previous_layer) {
	double sum = 0.0;
	// sum the prev layer's outputs (which are inputs)
	// include bias node from the prev layer
	for (unsigned n = 0; n < previous_layer.size(); ++n) {
		sum += previous_layer[n].GetOutputVal() * previous_layer[n].output_weights[my_index].weight;
	}

	output_value = Neuron::activationF(sum);
}
double Neuron::activationF(double x) {
	// tanh - range -1 to 1
	return tanh(x);
}
double Neuron::activationFDerivative(double x) {
	// tanh derivative
	return 1.0 - x * x;
}
void Neuron::SetOutputVal(double val) {
	output_value = val;
}

double Neuron::GetOutputVal() const {
	return output_value;
};
void Neuron::calculateOutputGradients(double val) {
	double delta = val - output_value;
	gradient = delta * Neuron::activationFDerivative(output_value);
}
void Neuron::calculateHiddenGradients(const Layer &next_layer) {
	double dow = sumDOW(next_layer);
	gradient = dow * Neuron::activationFDerivative(output_value);
}
double Neuron::sumDOW(const Layer &next_layer) const {
	double sum = 0.0;
	for(unsigned n = 0; n < next_layer.size() - 1; ++n) {
		sum += output_weights[n].weight * next_layer[n].gradient;
	}
	return sum;
}
void Neuron::updateInputWeights(Layer &previous_layer) {
	for(unsigned n = 0; n < previous_layer.size(); ++n) {
		Neuron &neuron = previous_layer[n];
		double old_delta_weight = neuron.output_weights[my_index].delta_weight;

		double new_delta_weight = eta * neuron.GetOutputVal() + alpha * old_delta_weight;

		neuron.output_weights[my_index].delta_weight = new_delta_weight;
		neuron.output_weights[my_index].weight += new_delta_weight;
	}
}
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

/*********************** Network ************************/
NeuralNetwork::NeuralNetwork(std::vector<unsigned> &topology) { //constructor
	unsigned n_of_layers = topology.size();
	for (unsigned l = 0; l < n_of_layers; ++l){
		layers.push_back(Layer());
		unsigned n_of_outputs = (n_of_layers == (topology.size() - 1)) ? 1 : topology[l + 1];

		for (unsigned n = 0; n <= topology[l]; ++n){
			layers.back().push_back(Neuron(n_of_outputs, n));
			std::cout << "Made Neuron" << std::endl;
		}
		layers.back().back().SetOutputVal(1.0);
	}
}

void NeuralNetwork::Forward(const std::vector<double> &input) {
	assert(input.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < input.size(); ++i) {
		layers[0][i].SetOutputVal(input[i]);
	}

	// Forward propagate
	for (unsigned l = 1; l < layers.size(); ++l) {
		Layer &previous_layer = layers[l - 1];

		for (unsigned n = 0; n < layers[l].size() - 1; ++n) {
			layers[l][n].Forward(previous_layer);
		}
	}
}
void NeuralNetwork::Backprop(const std::vector<double> &target) {
	// Calc overall net error (root mean square error of output neuron)
	Layer &output_layer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
		double delta = target[n] - output_layer[n].GetOutputVal();
		error += delta * delta;
	}
	error /= output_layer.size() - 1; // average error squared
	error = sqrt(error); // RMS

	// Calc output layer gradients
	for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
		output_layer[n].calculateOutputGradients(target[n]);
	}
	
	// Calc gradients on hidden layer
	for (unsigned l = layers.size() - 2; l > 0; --l) {
		Layer &hidden_layer = layers[l];
		Layer &next_layer = layers[l + 1];

		for (unsigned n = 0; n < hidden_layer.size(); ++n) {
			hidden_layer[n].calculateHiddenGradients(next_layer);
		}
	}
	
	// For all layers from outputs to first hidden layer
	// update Connection weights
	for (unsigned l = layers.size() - 1; l > 0; --l) {
		Layer &layer = layers[l];
		Layer &previous_layer = layers[l - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(previous_layer);
		}
	}
	
}
void NeuralNetwork::Results(std::vector<double> &result) const {
	result.clear();

	for(unsigned n = 0; n < layers.back().size() - 1; ++n) {
		result.push_back(layers.back()[n].GetOutputVal());
	}
}
