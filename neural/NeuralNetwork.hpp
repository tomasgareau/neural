//
//  NeuralNetwork.hpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <memory>
#include <vector>
#include "Neuron.hpp"
#include "Util.hpp"

using Util::Link;

class NeuralNetwork
{
private:
    std::vector<std::vector<std::unique_ptr<Neuron>>> _layers;
    std::vector<std::shared_ptr<Link>> _input;
    std::vector<std::shared_ptr<Link>> _output;
    
public:
    NeuralNetwork( std::vector<int> layer_sizes );
    
    void input( std::vector<double> data );
    int get_output();
    
    void feedforward();
    void stochastic_gradient_descent( std::vector<Util::DataPair>& training_data, int epochs, int mini_batch_size, double eta, std::vector<Util::DataPair>& test_data );
    void update_mini_batch( std::vector<Util::DataPair>::iterator start, std::vector<Util::DataPair>::iterator stop, double eta );
};

#endif /* NeuralNetwork_hpp */
