#include "neuron.h"
#include <iostream>
#include <cmath>

Neuron::Neuron(unsigned num_outputs, unsigned index)
{
  for(unsigned i = 0; i < num_outputs; i++)
  {
    output_weights.push_back(Connection());
    output_weights.back().weight = random_weight();
  }

  my_index = index;

  cout << "Neuron created\n";
}

void Neuron::feed_forward(Layer &prev_layer) // const?
{
  double sum = 0.0;

  // sum previous layers outputs
  // and include the weighted biased node.

  for(unsigned n = 0; n < prev_layer.size(); ++n)
  { // for each node in the previous layer.
    sum += prev_layer[n].get_output_value() *
      prev_layer[n].output_weights[my_index].weight;
  } // add up sum of the previous layers with their weights taken into consideraion.
  output_value = Neuron::transfer_function(sum);
}

double Neuron::transfer_function(double value)
{
  // tanh - output range [-1.0..1.0]
  return tanh(value);
}

double Neuron::transfer_function_derivate(double value)
{
    return (1.0 - (value * value));
}

// calculate the gradient of the output neurons
void Neuron::calc_output_gradients(double target_value)
{
  double delta = target_value - output_value;
  gradient = delta * Neuron::transfer_function_derivate(output_value);
}

//calculate the gradient of the hidden neurons. (hidden layers)
void Neuron::calc_hidden_gradients(const Layer &next_layer)
{
  double dow = sum_DOW(next_layer);
  gradient = dow * Neuron::transfer_function_derivate(output_value);
}

// using this for calculating previous layers.
double Neuron::sum_DOW(const Layer &next_layer) const
{
  double sum;
  for(unsigned n = 0; n < next_layer.size() - 1; ++n)
  {
    sum += output_weights[n].weight * next_layer[n].gradient;
  }
  return sum;
}


void Neuron::update_input_weights(Layer &prev_layer)
{
  for(unsigned n = 0; n < prev_layer.size(); ++n)
  {
    Neuron &neuron = prev_layer[n];
    double old_weight = neuron.output_weights[my_index].delta_weight;

    // eta = learning rate
    double new_delta_weight =
      eta
      * neuron.get_output_value()
      * gradient
      + alpha
      * old_weight;

      neuron.output_weights[my_index].delta_weight = new_delta_weight;
      neuron.output_weights[my_index].weight += new_delta_weight;
  }
}
