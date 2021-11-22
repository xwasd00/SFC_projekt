#include "madaline.hpp"

//adaline:  https://pabloinsente.github.io/the-adaline

//https://www.tutorialspoint.com/artificial_neural_network/artificial_neural_network_supervised_learning.htm


using namespace std;

int main(int argc, char const *argv[]) {
	vector<unsigned> topology = {3, 2, 1};
	vector<double> input = {0.1, 0.2, 0.3};
	vector<double> results;
	
	// NeuralNetwork n(topology);
	Madaline m(topology);
	m.print_network();
	m.forward(input);
	m.get_result(results);

	cout << "results: ";
	for(unsigned i = 0; i < results.size(); ++i) {
		cout << results[i] << "  ";
	}
	cout << endl;
	return 0;
}