//
//  Neuron.cpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#include "Neuron.hpp"

Neuron::Neuron( std::shared_ptr<std::vector<std::shared_ptr<Link>>> prev, std::shared_ptr<std::vector<std::shared_ptr<Link>>> next )
{
    _bias = Util::distribution( Util::generator );
    _prev = prev;
    _next = next;
}

void Neuron::add_link( std::shared_ptr<Link> link, Util::LinkDirection direction )
{
    if ( direction == Util::LinkDirection::PREV )
    {
        _prev->push_back( link );
    }
    else if ( direction == Util::LinkDirection::NEXT )
    {
        _next->push_back( link );
    }
    else {
        throw std::invalid_argument( "Direction is not a valid enum value from Util::LinkDirection" );
    }
}

double Neuron::_get_input()
{
    double input = 0.0;
    for ( std::shared_ptr<Link> link : *_prev )
    {
        input += link->output;
    }
    return input;
}

double Neuron::_get_output()
{
    return Util::sigmoid( _get_input() );
}

void Neuron::propagate_output()
{
    double activation = _get_output();
    for ( std::shared_ptr<Link> link : *_next )
    {
        link->output = activation * link->weight;
    }
}
