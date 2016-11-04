//
//  NeuralNetwork.cpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork( std::vector<int> layer_sizes )
{
    _layers.resize( layer_sizes.size() );
    for ( int i = 0; i < layer_sizes.size(); ++i )
    {
        for ( int j = 0; j < layer_sizes[i]; ++j )
        {
            _layers[i].push_back( std::make_unique<Neuron>() );
        }
    }
    
    // set the input links for the first layer
    for ( int i = 0; i < _layers[0].size(); ++i )
    {
        _input.push_back( std::make_shared<Link>() );
        _input.back()->weight = 1.0;
        _layers[0][i]->add_link( _input.back(), Util::LinkDirection::PREV );
    }
    
    // set the output links for the last layer
    for ( int i = 0; i < _layers.back().size(); ++i )
    {
        _output.push_back( std::make_shared<Link>() );
        _output.back()->weight = 1.0;
        _layers.back()[i]->add_link( _output.back(), Util::LinkDirection::NEXT );
    }
    
    // set up links between adjacent layers
    for ( int i = 0; i < _layers.size() - 1; ++i )
    {
        std::shared_ptr<Link> link;
        for ( std::unique_ptr<Neuron>& neuron : _layers[i] )
        {
            for ( std::unique_ptr<Neuron>& next_neuron : _layers[ i + 1 ] )
            {
                link = std::make_shared<Link>();
                neuron->add_link( link, Util::LinkDirection::NEXT );
                next_neuron->add_link( link, Util::LinkDirection::PREV );
            }
        }
    }
}

void NeuralNetwork::input( std::vector<double> data )
{
    if ( data.size() != _input.size() )
    {
        throw std::invalid_argument( "Input vector size does not match the size of the input layer." );
    }
    
    for ( int i = 0; i < data.size(); ++i )
    {
        _input[i]->output = data[i];
    }
}

int NeuralNetwork::get_output()
{
    double max_val = -1;
    int max_index = -1;
    for ( int i = 0; i < _output.size(); ++i )
    {
        if (_output[i]->output > max_val )
        {
            max_val = _output[i]->output;
            max_index = i;
        }
    }
    return max_index;
}

void NeuralNetwork::feedforward()
{
    for ( std::vector<std::unique_ptr<Neuron>>& layer : _layers)
    {
        for ( std::unique_ptr<Neuron>& neuron : layer )
        {
            neuron->propagate_activation();
        }
    }
}

void NeuralNetwork::stochastic_gradient_descent( std::vector<Util::DataPair>& training_data, int epochs, int mini_batch_size, double eta, std::vector<Util::DataPair>& test_data)
{
    time_t start, end;
    long num_cases = training_data.size();
    for ( int i = 0; i < epochs; ++i )
    {
        std::cout << "Epoch " << i << " starting... " << std::flush;
        time( &start );
        std::shuffle( training_data.begin(), training_data.end(), Util::generator );
        for ( int j = 0; j < num_cases; j += mini_batch_size )
        {
            std::vector<Util::DataPair>::iterator start = training_data.begin() + j;
            std::vector<Util::DataPair>::iterator stop = start + mini_batch_size;
            update_mini_batch( start, stop, eta );
        }
        
        int correct = 0;
        int total = static_cast<int>( test_data.size() );
        int output;
        if ( total != 0 )
        {
            for ( Util::DataPair test : test_data )
            {
                input( std::get<0>( test ) );
                feedforward();
                output = get_output();
                std::vector<double> tmp = std::get<1>( test );
                int expected_output = static_cast<int>( max_element( tmp.begin(), tmp.end() ) - tmp.begin() );
                if ( output == expected_output )
                {
                    ++correct;
                }
            }
        }
        time( &end );
        std::cout << "Finished epoch in " << difftime( end, start ) << " seconds. Performance: " << correct << " / " << total << "\n" << std::flush;
    }
}

void NeuralNetwork::update_mini_batch( std::vector<Util::DataPair>::iterator start, std::vector<Util::DataPair>::iterator stop, double eta)
{
    std::vector<double> expected_output;
    for ( std::vector<Util::DataPair>::iterator it = start; it != stop; ++it )
    {
        // set the input activation and feed propagate forward
        input( std::get<0>( *it ) );
        expected_output = std::get<1>( *it );
        feedforward();
        
        // calculate the output error
        for ( int i = 0; i < _output.size(); ++i )
        {
            // error = (current - expected) * sigmoid_prime(current)
            _output[i]->error = _output[i]->output - expected_output[i];
        }
        
        // backpropagate the error
        for ( int i = static_cast<int>(_layers.size() - 1); i >= 0; --i )
        {
            for ( std::unique_ptr<Neuron>& neuron : _layers[i] )
            {
                neuron->backprop();
            }
        }
    }
    
    // gradient descent
    double mini_batch_size = static_cast<double>( stop - start );
    for ( std::vector<std::unique_ptr<Neuron>>& layer : _layers )
    {
        for ( std::unique_ptr<Neuron>& neuron : layer )
        {
            neuron->gradient_descent( eta, mini_batch_size );
        }
    }
}
