#include <vector>
#include "net.cpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

int main()
{
    vector<unsigned> topology;

    string tmp_string;
    unsigned tmp_unsigned;

    vector<double> input_vals,results_vals,target_vals;

    cout << "NN1 test program...\n";
    cout << "loading input from file\n";

    ifstream file;
    file.open("trainingdata.txt");

    // read topology.
    getline(file, tmp_string);
    cout << tmp_string;
    stringstream stream(tmp_string);

    while(1) // read the net layout from input file.
    {
      stream >> tmp_unsigned;
      topology.push_back(tmp_unsigned);
      if(!stream)
        break;
    }

    cout << "Net size =" << topology.size() << "\n";
    for(int i = 0; i < topology.size(); i++)
      cout << "Layer(" << i << ") is composed of: " << topology.at(i) << " Neurons\n";


    Net my_net(topology); // create net lookling like the template specified by topology
    double tmp_double;

    while(!file.eof()) // while we can still read the file. fetch all input and outputs.
    { // while there is still more to read.
      input_vals.clear(); // clear all vectors storing data about the current run.
      results_vals.clear();
      target_vals.clear();

      // take to input.
      cout << "inputs: ";
      for(int i = 0; i < topology.at(0); i++)
      {
        file >> tmp_double;
        cout << tmp_double;
        input_vals.push_back(tmp_double);
      }
      cout << "\n";

      my_net.feed_forward(input_vals); // feed forward

      my_net.get_results(results_vals); // recieved the results.

      cout << "results: ";
      for(int i = 0; i < results_vals.size(); i++)
      {
        cout << results_vals.at(i);
      }
      cout << "\n";

      // take to target
      cout << "Targets: ";
      for(int i = 0; i < topology.back(); i++)
      {
        file >> tmp_double;
        cout << tmp_double;
        target_vals.push_back(tmp_double);
      }
      cout << "\n----------\n";

      my_net.back_propagation(target_vals); // evaluate how well the net operated.
    }
}
