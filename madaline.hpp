#include "adaline.hpp"
#include <cfloat>
#include <fstream>
#include <sstream>
#include <string>


struct t_Sample {
	std::vector<double> input;
	std::vector<double> desired_output;
};

class Madaline {
public:
	Madaline(const double &mi, const double &eps);
	void construct_topology(const std::vector<unsigned> &topology);
	void construct_topology(const std::string &file);
	void forward(const std::vector<double> &input);
	void update_input(const std::vector<double> &input);
	void forward();
	void get_result(std::vector<double> &results);
	void train(std::vector<t_Sample> &train_data, const double &threshold, const unsigned &max_iterations, const unsigned &debug_level);
	void update_network(const t_Sample &train_sample);
	void partial_forward(const unsigned layer_index, std::vector<double> &output);
	double error(const std::vector<double> &d, const std::vector<double> &y);
	void load_data(std::vector<t_Sample> &train_data,const std::string &train_file);
	void save_weights(std::string &save_file);
	void load_weights(std::string &load_file);
	~Madaline();

	//debug functions
	void print_network();
	void print_input();
	void print_output();
	void print_response_on_data(const std::vector<t_Sample> &train_data, const unsigned &debug_level);

private:
	std::vector<Layer> c_layers;
	double c_net_error;
	double c_eps = 0.1;
	double c_mi = 0.5;
};