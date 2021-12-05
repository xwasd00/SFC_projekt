#include "madaline.hpp"

using namespace std;

Madaline::Madaline(const double &mi, const double &eps) {
	c_mi = mi;
	c_eps = eps;
}

void Madaline::construct_topology(const string &file) {
	vector<unsigned> topology;
	fstream f;
	f.open(file, ios::in);
	if (f.is_open()) {
    	string tp;
    	getline(f, tp);
    	stringstream ss(tp);
    	unsigned n;
    	while(ss >> n) {
    		topology.push_back(n);
    	}
    }

	this->construct_topology(topology);
}

void Madaline::construct_topology(const vector<unsigned> &topology) {
	c_layers.clear();
	for(unsigned l = 0; l < topology.size(); ++l) {
		c_layers.push_back(Layer());
		unsigned layer_size = topology[l];
		for(unsigned n = 0; n < layer_size; ++n) {
			// first layer is input layer
			unsigned n_of_inputs = (l == 0) ? 0 : topology[l - 1];
			c_layers[l].push_back(Adaline(n, n_of_inputs, c_mi, c_eps));
		}
	}
}

void Madaline::forward(const vector<double> &input) {
	this->update_input(input);
	this->forward();
}

void Madaline::update_input(const vector<double> &input) {
	assert(input.size() == c_layers[0].size());
	
	// first layer - input layer
	// set output of neurons to input values
	for(unsigned n = 0; n < input.size(); ++n) {
		c_layers[0][n].set_output(input[n]);
	}
}

void Madaline::forward() {

	// all layers except input layer
	for(unsigned l = 1; l < c_layers.size(); ++l) {
		Layer &this_layer = c_layers[l];
		Layer &previous_layer = c_layers[l - 1];
		
		// propagate for each neuron in this layer
		for(unsigned n = 0; n < this_layer.size(); ++n) {
			this_layer[n].forward(previous_layer);
		}
	}
}

void Madaline::get_result(vector<double> &results) {
	results.clear();
	Layer &output_layer = c_layers.back();
	for(unsigned n = 0; n < output_layer.size(); ++n) {
		double neuron_output = output_layer[n].get_output();
		results.push_back(neuron_output);
	}
}

void Madaline::train(vector<t_Sample> &train_data, const double &threshold, const unsigned &max_iterations, const unsigned &debug_level) {
	assert(train_data.size() != 0);
	double sum_error = 0;
	double average_error = 0;
	unsigned iterations = 0;
	vector<double> results;

	if (debug_level > 0) {
		cout << "training..." << endl;
	}

	do {
		for (unsigned i = 0; i < train_data.size(); ++i) {
			this->update_network(train_data[i]);
			/*
			this->forward(train_data[i].input);
			this->get_result(results);
			c_net_error = this->error(train_data[i].desired_output, results);

			if(c_net_error > threshold) {
				this->update_network(train_data[i]);
			}
			*/
			
		}
		sum_error = 0;
		for(unsigned i = 0; i < train_data.size(); ++i) {
			this->forward(train_data[i].input);
			this->get_result(results);
			c_net_error = this->error(train_data[i].desired_output, results);
			sum_error += c_net_error; 
		}
		average_error = sum_error / (double) train_data.size();
		if (debug_level > 2) {
			cout << "average error: " << average_error << " iteration: " << iterations << endl;
		}
		++iterations;
	} while (average_error > threshold && iterations < max_iterations);

	if (debug_level > 0) {
		cout << "training ended" << endl;
	}
}

void Madaline::update_network(const t_Sample &train_sample) {
	vector<double> output;
	vector<double> altered_output;
	double neuron_net_error;

	

	// each layer, except 'input'
	for(unsigned l = 1; l < c_layers.size(); ++l) {

		// each neuron in layer 'l'
		for(unsigned n = 0; n < c_layers[l].size(); ++n) {
			Adaline &neuron = c_layers[l][n];
			Layer &previous_layer = c_layers[l - 1];

			this->forward(train_sample.input);
			this->get_result(output);
			c_net_error = this->error(train_sample.desired_output, output);
			
			// add perturbation and forward
			neuron.add_epsilon();
			this->partial_forward(l, altered_output);
			neuron.remove_epsilon();

			// compute network error with neuron perturbation
			neuron_net_error = this->error(train_sample.desired_output, altered_output);
			
			// update weights of neuron
			neuron.update_weights(previous_layer, c_net_error, neuron_net_error);
		}
	}
}

void Madaline::partial_forward(const unsigned layer_index, vector<double> &output) {
	for(unsigned l = layer_index; l < c_layers.size(); ++l) {
		Layer &this_layer = c_layers[l];
		Layer &previous_layer = c_layers[l - 1];
		
		// propagate for each neuron in this layer
		for(unsigned n = 0; n < this_layer.size(); ++n) {
			this_layer[n].forward(previous_layer);
		}
	}
	this->get_result(output);
}

double Madaline::error(const vector<double> &d, const vector<double> &y) {
	// sum squared output response error
	// i.e. (d_1 - y_1)^2 + (d_2 - y_2)^2 for network output vector [y_1, y_2]
	assert(d.size() == y.size());
	double sum = 0.0;
	for(unsigned i = 0; i < d.size(); ++i) {
		sum += (d[i] - y[i]) * (d[i] - y[i]);
	}
	return sum;
}

void Madaline::load_data(vector<t_Sample> &train_data, const string &train_file) {
	fstream file;
	file.open(train_file, ios::in);
	if (file.is_open()) {
    	string tp;
    	vector<double> tmp_data;
    	double d;

    	while(getline(file, tp)) {
    		stringstream ss(tp);
    		tmp_data.clear();
    		while (ss >> d) {
        		tmp_data.push_back(d);
    		}
    		if(tmp_data.size() == c_layers[0].size() + c_layers.back().size()){
    			t_Sample sample; 
    			for(unsigned i = 0; i < c_layers[0].size(); ++i) {
        			sample.input.push_back(tmp_data[i]);
        		}
        		for(unsigned o = c_layers[0].size(); o < tmp_data.size(); ++o) {
        			sample.desired_output.push_back(tmp_data[o]);
        		}
        		train_data.push_back(sample);
        	}
    	}
    }
    file.close();
}


void Madaline::save_weights(string &save_file){
	ofstream file(save_file);
	for(unsigned l = 0; l < c_layers.size(); ++l) {
		for(unsigned n = 0; n < c_layers[l].size(); ++n) {
			c_layers[l][n].save_weights(file);
		}
		file << endl;
	}
}

void Madaline::load_weights(string &load_file){
	fstream f;
	f.open(load_file, ios::in);
	if (f.is_open()) {
    	string s;
    	unsigned l = 0;
    	c_layers.clear();
    	while(getline(f, s)){
    		stringstream ss(s);
    		Layer layer;
    		unsigned index = 0;
    		unsigned n_of_inputs = (l == 0) ? 0 : c_layers.back().size();
    		while(true) {
    			Adaline neuron(index, n_of_inputs, c_mi, c_eps);
    			string tmp;
    			stringstream n;
    			while (ss >> tmp){
    				tmp += " ";
    				n << tmp;
    				if (tmp.find("]") != string::npos) {
    					break;
    				}
    			}
    			if(!neuron.load_weights(n)){
    				break;
    			}
    			layer.push_back(neuron);
    			index++;
    		}
    		c_layers.push_back(layer);
    		l++;
    	}
    }
}

Madaline::~Madaline() {}

/***************************  debug functions  *************************/
void Madaline::print_network() {
	cout << "TOPOLOGY:" << endl;
	for(unsigned l = 0; l < c_layers.size(); ++l) {
		cout << "Layer " << l;
		if (l == 0) {
			cout << "(input layer - has only biases)";
		}
		cout << ":" << endl;
		for(unsigned n = 0; n < c_layers[l].size(); ++n) {
			cout << "\t";
			c_layers[l][n].print_neuron();
		}
	}
}

void Madaline::print_output() {
	Layer &output_layer = c_layers.back();
	cout << "OUTPUT: [ ";
	for(unsigned n = 0; n < output_layer.size(); ++n) {
		cout << output_layer[n].get_output() << " ";
	}
	cout << "]" << endl;
}

void Madaline::print_input() {
	Layer &input_layer = c_layers[0];
	cout << "INPUT: [ ";
	for(unsigned n = 0; n < input_layer.size(); ++n) {
		cout << input_layer[n].get_output() << " ";
	}
	cout << "]" << endl;
}


void Madaline::print_response_on_data(const vector<t_Sample> &train_data, const unsigned &debug_level) {
	int wrong_count = 0;
	bool wrong_output = false;
	vector<double> results;
	
	cout << "---------------------------------------------" << endl;
	cout << "RESPONSE:" << endl;
	if(debug_level > 1){
		this->print_network();
	}
	cout << endl;
	cout << "DATA:" << endl;
	for (unsigned i = 0; i < train_data.size(); ++i) {
		this->forward(train_data[i].input);
		this->get_result(results);
		cout << "\t";
		this->print_input();
		cout << "\t";
		this->print_output();
		cout << "\tDESIRED: [ ";
		for(unsigned j = 0; j < train_data[i].desired_output.size(); ++j) {
			cout << train_data[i].desired_output[j] << " ";
			if(abs(train_data[i].desired_output[j] - results[j]) >= 0.5) {
				wrong_output = true;
				break;
			}
		}
		cout << "]" << endl << endl;

		if (wrong_output) {
			wrong_output = false;
			wrong_count++;
		}
	}
	cout << "Data size: " << train_data.size() << "  Wrong classified: " << wrong_count << endl;
	cout << "---------------------------------------------" << endl;
}