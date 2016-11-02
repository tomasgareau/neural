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
    std::vector<std::shared_ptr<Link>> _links;
    
public:
    NeuralNetwork( std::vector<int> layer_sizes );
    ~NeuralNetwork();
};

#endif /* NeuralNetwork_hpp */
