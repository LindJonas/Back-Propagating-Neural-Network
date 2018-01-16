#include "neuron.cpp"
#include <cmath>

using namespace std;

typedef vector<Neuron> Layer;

class Net
{
public:
    // for training
    Net(const vector<unsigned> &topology);
    // constructor takes vector where size determins the amount of layers, and contained values
    // determins amount of neurons in each layers.


    // functions for training.
    void feed_forward(const vector<double> &input_vals);
    void back_propagation(const vector<double> &target_vals);

    // function for operation after training.
    void get_results(vector<double> &results_vals);

private:
    //  Vector of layers that will make up the net.
    vector<Layer> m_layers;
    double m_error;
    double recent_avrage_error;
    double recent_average_smoothing_factor;
};
