#include <vector>
#include <cstdlib>
//#include "connection.cpp"
#include <vector>

#ifndef NEURON
#define NEURON

using namespace std;

struct Connection; // connections and their values
class Neuron; //

typedef vector<Neuron> Layer;


struct Connection
{
  double weight;
  double delta_weight;
};

class Neuron
{
public:

    Neuron(unsigned num_outputs, unsigned index);
    void set_ouput_value(double value) { output_value = value; }
    double get_output_value() { return output_value; }
    void feed_forward(Layer &prev_layer);
    void calc_output_gradients(double target_value);
    void calc_hidden_gradients(const Layer &next_layer);
    void update_input_weights(Layer &prev_layer);

private:

    static double transfer_function(double value);
    static double transfer_function_derivate( double value);

    // these can be experimented with.
    static const double eta = 0.15; // 0 -> 1
    // 0.0 slow learning
    // 0.2 medium learning
    // 1.0 aggressive learning (results may vary)
    static const double alpha = 0.5; // 0 -> 1
    // 0.0 no momentum
    // 0.5 moderate momentum

    static double random_weight() { return rand() / double(RAND_MAX); }
    double sum_DOW(const Layer &next_layer) const;
    double output_value;

    vector<Connection> output_weights;

    unsigned my_index;
    double gradient;

    // connection is not implemented yet.
};

#endif
