//
//  Neuron.cpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#include "Neuron.hpp"

Neuron::Neuron()
{
    _bias = Util::distribution( Util::generator );
    _prev = std::make_unique<std::vector<std::shared_ptr<Link>>>();
    _next = std::make_unique<std::vector<std::shared_ptr<Link>>>();
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

double Neuron::get_input()
{
    double input = 0.0;
    for ( std::shared_ptr<Link> link : *_prev )
    {
        input += link->output * link->weight;
    }
    return input + _bias;
}

double Neuron::get_output()
{
    return Util::sigmoid( get_input() );
}

void Neuron::propagate_activation()
{
    double activation = get_output();
    for ( std::shared_ptr<Link> link : *_next )
    {
        link->output = activation;
    }
}

void Neuron::backprop()
{
    double error = 0.0;
    for ( std::shared_ptr<Link> link : *_next )
    {
        error += link->weight * link->error;
    }
    error *= Util::sigmoid( get_output() );
    for ( std::shared_ptr<Link> link : *_prev )
    {
        link->error = error;
    }
}

void Neuron::gradient_descent( double eta, double mini_batch_size )
{
    for ( std::shared_ptr<Link> link : *_next )
    {
        link->weight -= ( eta / mini_batch_size ) * link->output * link->error;
    }
    _bias -= ( eta / mini_batch_size ) * (*_prev)[0]->error;
}
