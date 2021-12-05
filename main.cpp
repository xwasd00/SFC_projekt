#include "madaline.hpp"
#include <getopt.h>

using namespace std;

bool file_exist(char *path) {
	ifstream file(path);
	return file.is_open();
}


void print_help() {
	cout << "usage: ./main [-h] -g TOPOLOGY [-r TRAIN] [-t TEST] [-s SAVE] [-m MI] [-e EPS] [-p THRESHOLD] [-i I] [-d LEVEL]" << endl;
	cout << "or ./main [-h] -l LOAD [-r TRAIN] [-t TEST] [-s SAVE] [-m MI] [-e EPS] [-p THRESHOLD] [-i I] [-d LEVEL]" << endl;
	cout << endl;
	cout << "optional arguments:" << endl;
	cout << "  -h, --help                show this help message and exit" << endl;
	cout << "  -r, --train TRAIN         train network with training data from the file TRAIN" << endl;
	cout << "  -t, --test TEST           test network with test data from the file TEST" << endl;
	cout << "  -s, --save SAVE           file, where trained network is stored" << endl;
	cout << "  -m, --mi MI               learning coeficient (default 0.6)" << endl;
	cout << "  -e, --eps EPS             perturbation - Delta s (default 0.01)" << endl;
	cout << "  -p, --threshold THRESHOLD" << endl;
	cout << "                            minimal average error of training" << endl;
	cout << "                            training will stop if net error is less than threshold (default 0.1)" << endl;
	cout << "  -i, --iterations I        number of iterations over train data (default 10000)" << endl;
	cout << "  -d, --debug LEVEL         level of debugging info:" << endl;
	cout << "                            0: no debugging info" << endl;
	cout << "                            1: level 0 + response to train data" << endl;
	cout << "                            2: level 1 + print network" << endl;
	cout << "                            3: level 2 + print average error with every iteration in training" << endl;
	cout << "                               and print network on load" << endl;
	cout << endl;
	cout << "required named arguments:" << endl;
	cout << "  -g, --topology TOPOLOGY" << endl;
	cout << "                            file, where is stored topology of network" << endl;
	cout << "  -l, --load LOAD           file, where is stored network, this will load that network" << endl;
	return;
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

	// get option from getopt_long 
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
		cerr << "cannot load both TOPOLOGY file and LOAD file, choose one" << endl;
		return 2;
	}

	Madaline m(mi, eps);

	// load topology -> initialize with random weights
	if (topology_file.size() != 0) {
		m.construct_topology(topology_file);
		if (debug_level > 2) {
			cout << "LOADED NETWORK ";
			m.print_network();
		}
	}

	// load network -> load weights from file
	if (w_load_file.size() != 0){
		m.load_weights(w_load_file);
		if (debug_level > 2) {
			cout << "LOADED NETWORK ";
			m.print_network();
		}
	}

	// load training data and train network
	if (train_file.size() != 0) {

		// load training data
		vector<t_Sample> train_data;
		m.load_data(train_data, train_file);
		if (debug_level > 0) {
			cout << endl;
			cout << "TRAINING DATA:" << endl;
			m.print_response_on_data(train_data, debug_level);
		}

		// train network
		m.train(train_data, threshold, iterations, debug_level);
		if (debug_level > 0) {
			cout << endl;
			cout << "TRAINING DATA:" << endl;
			m.print_response_on_data(train_data, debug_level);
		}
	}

	// load test data and print response
	if (test_file.size() != 0) {
		vector<t_Sample> test_data;
		m.load_data(test_data, test_file);
		cout << endl;
		cout << "TEST DATA:" <<endl;
		m.print_response_on_data(test_data, debug_level);
	}

	// save to file
	if (w_save_file.size() != 0) {
		m.save_weights(w_save_file);
	}

	return 0;
}