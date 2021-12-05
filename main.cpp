#include "madaline.hpp"
#include <getopt.h>

//https://www.slideshare.net/infobuzz/adaline-madaline
//https://pdfs.semanticscholar.org/ea91/de67f2f2d8635349cdbde54cc8c7e43f13c5.pdf
//http://ziyang.eecs.umich.edu/~dickrp/iesr/lectures/widrow90sep-present.pdf
//https://www.cmpe.boun.edu.tr/~ethem/files/papers/annsys.pdf
//https://link.springer.com/article/10.1007/s00521-009-0298-3

//datasets? mehrotra.zip, proben1.zip

// getopt: https://codeyarns.com/2015/01/30/how-to-parse-program-options-in-c-using-getopt_long/

using namespace std;

bool file_exist(char *path) {
	ifstream file(path);
	return file.is_open();
}


void print_help() {
	cout << "TODO: help" << endl;
}

int main(int argc, char **argv) {
	unsigned debug_level = 0;
	double eps = 0.01;
	double mi = 0.6;
	double threshold = 0.1;
	unsigned iterations = 10000;
	string topology_file;
	string w_save_file;
	string w_load_file;
	string test_file;
	string train_file;
	
	/******************** argument parsing *********************************/
	//short options
	const char* const short_opts = ":m:e:l:s:g:t:r:p:i:d:h";
	// long options
	const option longopts[] = {
		{"help", no_argument, nullptr, 'h'},
		{"mi", required_argument, nullptr, 'm'},
		{"eps", required_argument, nullptr, 'e'},
		{"load", required_argument, nullptr, 'l'},
		{"save", required_argument, nullptr, 's'},
		{"topology", required_argument, nullptr, 'g'},
		{"test", required_argument, nullptr, 't'},
		{"train", required_argument, nullptr, 'r'},
		{"threshold", required_argument, nullptr, 'p'},
		{"iterations", required_argument, nullptr, 'i'},
		{"debug", required_argument, nullptr, 'd'}
	};
	int option;

	// získání možností z getopt_long 
	while((option = getopt_long(argc, argv, short_opts, longopts, nullptr)) != -1){
		switch(option){

			// ./main --eps <eps>
        	case 'e':
				eps = stod(optarg);
				break;

			// ./main --mi <mi>
			case 'm':
				mi = stod(optarg);
				break;
			
			// ./main --load <w_load_file>
			case 'l':
				if (!file_exist(optarg)) {
        			cerr << "File " << optarg << " not found" << endl; 
        			return 2;          
				}
				w_load_file = optarg;
				break;

			// ./main --save <w_save_file>
			case 's':
				if (!file_exist(optarg)) {
        			cerr << "File " << optarg << " not found" << endl;  
        			return 2;         
				}
				w_save_file = optarg;
				break;

			// ./main --topology <topology_file>
			case 'g':
				if (!file_exist(optarg)) {
        			cerr << "File " << optarg << " not found" << endl;  
        			return 2;         
				}
				topology_file = optarg;
				break;

			// ./main --test <test_file>
			case 't':
				if (!file_exist(optarg)) {
        			cerr << "File " << optarg << " not found" << endl;  
        			return 2;         
				}
				test_file = optarg;
				break;

			// ./main --train <train_file>
			case 'r':
				if (!file_exist(optarg)) {
        			cerr << "File " << optarg << " not found" << endl;  
        			return 2;         
				}
				train_file = optarg;
				break;

			// ./main --threshold <threshold>
			case 'p':
				threshold = stod(optarg);
				break;

			// ./main --iterations <iterations>
			case 'i':
				iterations = stoi(optarg);
				break;

			// ./main --debug <level>
			case 'd':
				debug_level = stoi(optarg);
				break;
			
			case ':':
				cerr << "Option -" << (char)optopt << " needs value." << endl;
				return 2;
				break;

			case 'h': 
				print_help();
				return 0;

        	case '?': // Unrecognized option
        	default:
        		cerr << "Unrecognized option: " << (char)optopt << endl;
            	print_help();
            	return 2;
            	break;
		}
	}
	/******************* end argument parsing ******************************/

	// only one of them can be loaded
	if (!((w_load_file.size() == 0) ^ (topology_file.size() == 0))) {
		cerr << "cannot load both topology_file and w_load_file, choose one" << endl;
		return 2;
	}

	Madaline m(mi, eps);

	if (topology_file.size() != 0) {
		m.construct_topology(topology_file);
		if (debug_level > 2) {
			cout << "LOADED NETWORK ";
			m.print_network();
		}
	}
	if (w_load_file.size() != 0){
		m.load_weights(w_load_file);
		if (debug_level > 2) {
			cout << "LOADED NETWORK ";
			m.print_network();
		}
	}

	if (train_file.size() != 0) {
		// load training data
		vector<t_Sample> train_data;
		m.load_data(train_data, train_file);
		if (debug_level > 0) {
			m.print_response_on_train_data(train_data, debug_level);
		}

		// train
		m.train(train_data, threshold, iterations, debug_level);
		if (debug_level > 0) {
			m.print_response_on_train_data(train_data, debug_level);
		}
	}

	if (test_file.size() != 0) {
		vector<t_Sample> test_data;
		m.load_test_data(test_data, test_file);
		m.print_response_on_data(test_data);
	}

	if (w_save_file.size() != 0) {
		m.save_weights(w_save_file);
	}

	return 0;
}