#include "adaline.hpp"

class Madaline {
public:
	Madaline(std::vector<unsigned> &topology);
	void forward(const std::vector<double> &input);
	void get_result(std::vector<double> &results);
	void back_propagate(const std::vector<double> &target_results);
	~Madaline();

	//debug functions
	void print_network();

private:
	std::vector<Layer> c_layers;
	std::vector<double> c_net_error;
};