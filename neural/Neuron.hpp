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
    std::shared_ptr<std::vector<std::shared_ptr<Link>>> _prev;
    std::shared_ptr<std::vector<std::shared_ptr<Link>>> _next;
    
    double _get_input();
    double _get_output();
    
public:
    Neuron( std::shared_ptr<std::vector<std::shared_ptr<Link>>> prev, std::shared_ptr<std::vector<std::shared_ptr<Link>>> next );
    virtual ~Neuron();
    
    void add_link( std::shared_ptr<Link> link, Util::LinkDirection direction );
    void propagate_output();
};

#endif /* Neuron_hpp */
