//
//  Neuron.hpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp

#include <vector>
#include <memory>
#include "Util.hpp"

using Util::Link;

class Neuron
{
private:
    double _bias;
    std::unique_ptr<std::vector<std::shared_ptr<Link>>> _prev;
    std::unique_ptr<std::vector<std::shared_ptr<Link>>> _next;
    
public:
    Neuron();
    
    void add_link( std::shared_ptr<Link> link, Util::LinkDirection direction );
    void propagate_activation();
    void backprop();
    void gradient_descent( double eta, double mini_batch_size );
    double get_input();
    double get_output();
};

#endif /* Neuron_hpp */
