#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <assert.h>

class Adaline;
typedef std::vector<Adaline> Layer;

class Adaline {
public:
	Adaline(const unsigned index, const unsigned n_of_inputs);
	void initialize_weights(const unsigned n_of_inputs);
	void forward(const Layer &input_layer);
	double get_output() const;
	void set_output(const double val);
	double activation_function(const double val);
	void add_epsilon();
	void remove_epsilon();
	void update_weights(const Layer &previous_layer, const double net_e, const double neuron_net_e);
	~Adaline();

	// debug functions
	void print_neuron();
private:
	unsigned c_index;
	double c_output;
	std::vector<double> c_weights;
	double c_epsilon = 0.1;
	double c_mi = 0.6;
	bool c_add_epsilon = false;
};