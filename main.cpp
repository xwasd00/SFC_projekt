#include "madaline.hpp"

//adaline:  https://pabloinsente.github.io/the-adaline
//https://www.slideshare.net/infobuzz/adaline-madaline
//https://pdfs.semanticscholar.org/ea91/de67f2f2d8635349cdbde54cc8c7e43f13c5.pdf
//http://ziyang.eecs.umich.edu/~dickrp/iesr/lectures/widrow90sep-present.pdf
//https://www.cmpe.boun.edu.tr/~ethem/files/papers/annsys.pdf
//https://link.springer.com/article/10.1007/s00521-009-0298-3

//simple dataset? https://codereview.stackexchange.com/questions/254212/simple-dataset-class-in-c
//other datasets? mehrotra.zip, proben1.zip

using namespace std;

int main(int argc, char const *argv[]) {
	// TODO - argumenty:
	// mi, eps, w_load_file, w_save_file, topology_file, test_file, train_file, debug level
	double eps = 0.01;
	double mi = 0.6;
	string w_save_file = "w.test";
	string w_load_file = "w.test";
	
	//string topology_file = "xor-topology.txt";
	string topology_file = "B8-Spiral-topology.txt";

	//string test_file = "xor-test.txt";
	string test_file = "B8-Spiral.dta";

	//string train_file = "xor.txt";
	string train_file = "B8-Spiral.dta";



	Madaline m(topology_file, mi, eps);

	vector<t_Sample> train_data;
	m.load_data(train_data, train_file);
	
	vector<t_Sample> test_data;
	m.load_data(test_data, test_file);

	cout << "train data" << endl;
	//m.print_response_on_train_data(train_data);
	m.print_response_on_data(train_data);
	
	cout << "training" << endl;
	m.train(train_data, 0.1, 10000);


	cout << "test data" << endl;
	//m.print_response_on_train_data(test_data);
	m.print_response_on_data(test_data);

	return 0;
}