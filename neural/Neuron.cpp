#include "Neuron.h"

Neuron::Neuron()
{
    _next_layer = nullptr;
    _bias = distribution( generator );
    _activation = 0.0;
    _error = 0.0;
}

void Neuron::init( vector<Neuron*>* next_layer )
{
    _next_layer = next_layer;
    if ( _next_layer != nullptr )
    {
        for ( int i = 0; i < next_layer->capacity(); ++i )
        {
            _weights.push_back( distribution( generator ) );
        }
    }
}

void Neuron::input( double val )
{
    _activation += sigmoid( val + _bias );
}

double Neuron::get_activation()
{
    return _activation;
}

double Neuron::get_error()
{
    return _error;
}

void Neuron::set_error( double val )
{
    _error = val;
}

void Neuron::feedforward()
{
    if ( _next_layer != nullptr )
    {
        for ( int i = 0; i < _next_layer->size(); ++i )
        {
            (*_next_layer)[i]->input( _weights[i] * _activation );
        }
    }
}

void Neuron::backprop()
{
    if ( _next_layer != nullptr )
    {
        _error = 0;
        double activation_factor = sigmoid_prime( get_activation() );
        for ( int i = 0; i < (*_next_layer).size(); ++i )
        {
            _error += (*_next_layer)[i]->get_error() * _weights[i];
        }
        _error *= activation_factor;
    }
}

void Neuron::gradient_descent( double eta, long mini_batch_size )
{
    if ( _next_layer != nullptr ) 
    {
        for ( int i = 0; i < (*_next_layer).size(); ++i )
        {
            _weights[i] -= ( eta / mini_batch_size ) * get_activation() * (*_next_layer)[i]->get_error();
        }
    }

    _bias -= ( eta / mini_batch_size ) * get_error();
}

