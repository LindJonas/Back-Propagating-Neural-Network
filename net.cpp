#include "net.h"
#include <cassert>


Net::Net(const vector<unsigned> &topology)
{
  unsigned num_layers = topology.size();

  for(unsigned i = 0; i < num_layers; i++)
  { // for each layer...
    m_layers.push_back(Layer()); // create new layer and push it to the layer vector.

    unsigned num_outputs;

    if(i == topology.size() - 1)
      num_outputs = i;
    else
      num_outputs = topology[i + 1];

    for(unsigned j = 0; j <= topology[i]; j++) // j < topology[i] if we want a weighted node.
    { // <= adds a biased neuron. (const neuron)
      m_layers.back().push_back(Neuron(num_outputs,j));
    }
    // force the weighted node to be one 1
    m_layers.back().back().set_ouput_value(1.0);
    // this should not be changed at all
  }
}

void Net::feed_forward(const vector<double> &input_vals)
{
    //error checking should be here.
    // check if the size of the input value vector is possible to fit in the
    // first layer (input layer)
    assert(input_vals.size() == m_layers[0].size() - 1);

    for(unsigned i = 0; i < input_vals.size(); ++i)
    {
      m_layers[0][i].set_ouput_value(input_vals[i]);
    }

    // ( call in order every layer and every neuron in said layer and call function feed_forward).
    for(unsigned layer_num = 1; layer_num < m_layers.size(); ++layer_num)
    {
      Layer &prev_layer = m_layers[layer_num - 1]; // trouble here

      for(unsigned n = 0; n < m_layers[layer_num].size() - 1; ++n)
      {
        m_layers[layer_num][n].feed_forward(prev_layer);
        // in layer_num at neuron n
      }
    }
}

void Net::back_propagation(const vector<double> &target_vals)
{
  // calculate overall net error
  Layer &output_layer = m_layers.back();
  m_error = 0.0;

  for( unsigned n = 0; n < output_layer.size() - 1; ++n)
  {
    double delta = target_vals[n] - output_layer[n].get_output_value();
  }

  m_error /= output_layer.size() - 1;
  m_error = sqrt(m_error);

  // calculate output layer gradients

  recent_avrage_error = (recent_avrage_error + recent_average_smoothing_factor + m_error)
      / (recent_average_smoothing_factor + 1.0);

  // calculate gradient on hidden layers

  for(unsigned n = 0; n < output_layer.size() - 1; ++n)
  {
    output_layer[n].calc_output_gradients(target_vals[n]);
  }

  for(unsigned layer_num = m_layers.size() - 2; layer_num > 0; -- layer_num)
  {
      Layer &hidden_layer = m_layers[layer_num];
      Layer &next_layer = m_layers[layer_num + 1];

      for (unsigned n = 0; n < hidden_layer.size(); ++n)
      {
        hidden_layer[n].calc_hidden_gradients(next_layer);
      }
  }

  for(unsigned layer_num = m_layers.size() - 1; layer_num > 0; --layer_num)
  {
    Layer &layer = m_layers[layer_num];
    Layer &prev_layer = m_layers[layer_num - 1];

    for(unsigned n = 0; n < layer.size() - 1; ++n)
    {
      layer[n].update_input_weights(prev_layer);
    }
  }

  // for all layer from output to first hidden layer
  // update connection weights.
}

void Net::get_results(vector<double> &results_vals)
{
    results_vals.clear();

    for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
      results_vals.push_back(m_layers.back()[n].get_output_value());
    }
}
