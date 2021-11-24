#include "madaline.hpp"

//adaline:  https://pabloinsente.github.io/the-adaline
//https://www.slideshare.net/infobuzz/adaline-madaline
//https://pdfs.semanticscholar.org/ea91/de67f2f2d8635349cdbde54cc8c7e43f13c5.pdf
//http://ziyang.eecs.umich.edu/~dickrp/iesr/lectures/widrow90sep-present.pdf
//https://www.cmpe.boun.edu.tr/~ethem/files/papers/annsys.pdf
//https://link.springer.com/article/10.1007/s00521-009-0298-3

using namespace std;

int main(int argc, char const *argv[]) {
	vector<unsigned> topology = {2, 2, 1};
	vector<t_Sample> train_data;

	t_Sample s1;
	s1.input = {0, 0};
	s1.desired_output = {0};
	train_data.push_back(s1);
	
	t_Sample s2;
	s2.input = {0, 1};
	s2.desired_output = {1};
	train_data.push_back(s2);

	t_Sample s3;
	s3.input = {1, 0};
	s3.desired_output = {1};
	train_data.push_back(s3);

	t_Sample s4;
	s4.input = {1, 1};
	s4.desired_output = {0};
	train_data.push_back(s4);


	Madaline m(topology);
	m.print_response_on_train_data(train_data);
	cout << "training" << endl;
	m.train(train_data);
	m.print_response_on_train_data(train_data);
	return 0;
}