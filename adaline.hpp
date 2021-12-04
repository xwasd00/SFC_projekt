#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <assert.h>
#include <fstream>
#include <sstream>


class Adaline;
typedef std::vector<Adaline> Layer;

class Adaline {
public:
	Adaline(const unsigned index, const unsigned n_of_inputs, const double &mi, const double &eps);
	Adaline(const unsigned index, const unsigned n_of_inputs);
	void initialize_weights(const unsigned n_of_inputs);
	void forward(const Layer &input_layer);
	double get_output() const;
	void set_output(const double val);
	double activation_function(const double val);
	void add_epsilon();
	void remove_epsilon();
	void update_weights(const Layer &previous_layer, const double net_e, const double neuron_net_e);
	void save_weights(std::ofstream &file);
	bool load_weights(std::stringstream &ss);
	~Adaline();

	// debug functions
	void print_neuron();
private:
	unsigned c_index;
	double c_output;
	double c_potential;
	std::vector<double> c_weights;
	double c_epsilon;
	double c_mi;
	bool c_add_epsilon = false;
};