#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <assert.h>

//https://www.youtube.com/watch?v=sK9AbJ4P8ao

class Neuron;
typedef std::vector<Neuron> Layer;

/********************** Connection ***********************/
class Connection {
public:
	Connection();
	double weight;
	double delta_weight;
private:
	double random_weight();
};

/************************ Neuron *************************/
class Neuron {
public:
	Neuron(unsigned n_of_outputs, unsigned index);
	void Forward(const Layer &previous_layer);
	void SetOutputVal(double val);
	double GetOutputVal() const;
	void calculateOutputGradients(double val);
	void calculateHiddenGradients(const Layer &next_layer);
	double sumDOW(const Layer &next_layer) const;
	void updateInputWeights(Layer &previous_layer);
private:
	static double eta;  // [0.0..1.0] overall training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
	double output_value;
	std::vector<Connection> output_weights;
	unsigned my_index;
	double gradient;
	static double activationF(double x);
	static double activationFDerivative(double x);
};

/*********************** Network ************************/
class NeuralNetwork {
public:
	NeuralNetwork(std::vector<unsigned> &topology);

	void Forward(const std::vector<double> &input);
	void Backprop(const std::vector<double> &target);
	void Results(std::vector<double> &result) const;
private:
	std::vector<Layer> layers;
	double error;
};