#include "adaline.hpp"
#include <cfloat>

struct t_Sample {
	std::vector<double> input;
	std::vector<double> desired_output;
};

class Madaline {
public:
	Madaline(const std::vector<unsigned> &topology);
	void forward(const std::vector<double> &input);
	void get_result(std::vector<double> &results);
	void train(const std::vector<t_Sample> &train_data);
	void update_network(const std::vector<double> &desired_results);
	void partial_forward(const unsigned layer_index, std::vector<double> &output);
	double error(const std::vector<double> &a, const std::vector<double> &b);
	~Madaline();

	//debug functions
	void print_network();
	void print_input();
	void print_output();
	void print_response_on_train_data(const std::vector<t_Sample> &train_data);

private:
	std::vector<Layer> c_layers;
	double c_net_error;
};