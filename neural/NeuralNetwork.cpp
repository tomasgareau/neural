#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork( vector<int>& sizes )
{
    for ( auto i : sizes )
    {
        _layers.push_back( vector<Neuron>( i ) );
    }
}

void NeuralNetwork::init()
{
    for ( int i = 0; i < _layers.size() - 1; ++i )
    {
         for ( auto neuron : _layers[i] )
         {
             neuron.init( &_layers[i+1] );
         }
    }

    for ( auto neuron : _layers.back() )
    {
        neuron.init( nullptr );
    }
}

void NeuralNetwork::stochastic_gradient_descent( Data& training_data, int epochs, int mini_batch_size, double eta, Data* test_data = nullptr )
{
    time_t start, end;
    int num_cases = training_data.size();
    for ( int i = 0; i < epochs; ++i )
    {
        std::cout << "Epoch " << i << " starting... " << std::flush;
        time( &start );
        training_data.shuffle();
        for ( int j = 0; j < num_cases; j += mini_batch_size )
        {
            DataIterator start = training_data.begin() + j;
            DataIterator stop = start + mini_batch_size;
            update_mini_batch( start, stop, eta );
        }

        int correct = 0;
        int total = test_data->size();
        if ( test_data != nullptr )
        {
            for ( auto test: *test_data )
            {
                this->input( std::get<0>( test ) );
                this->feedforward();
                int output = this->get_output_int();
                VectorD tmp = std::get<1>( test );
                int expected_output = max_element( tmp.begin(), tmp.end() ) - tmp.begin();
                if ( output == expected_output )
                {
                    ++correct;
                }
            }
        }
        time( &end );
        std::cout << "Finished epoch in " << difftime( end, start ) << " seconds. Performance: " << correct << " / " << total << std::endl;
    }
}

void NeuralNetwork::update_mini_batch( DataIterator start, DataIterator stop, double eta )
{
    VectorD expected_output;
    VectorD output_error;
    VectorD actual_output;

    for ( DataIterator it = start; it != stop; ++it )
    {
        // set the input activation
        input( std::get<0>( *it ) );
        expected_output = std::get<1>( *it );

        feedforward();

        Neuron tmp_neuron;
        // calculate the output error
        for ( int i = 0; i < _layers.back().size(); ++i )
        {
            tmp_neuron.set_error( ( expected_output[i] - Neuron::sigmoid( tmp_neuron.get_activation() ) ) * Neuron::sigmoid_prime( tmp_neuron.get_activation() ) );
        }

        // backpropagate the error
        for ( auto it = _layers.end() - 2; it > _layers.begin(); --it )
        {
            for ( auto neuron: *it )
            {
                neuron.backprop();
            }
        }
    }

    // gradient descent
    for ( auto vec: _layers )
    {
        for ( auto neuron: vec )
        {
            neuron.gradient_descent( eta, _layers.end() - _layers.begin() );
        }
    }
}

void NeuralNetwork::feedforward()
{
    // feedforward
    for ( auto layer: _layers )
    {
        for ( auto neuron: layer )
        {
            neuron.feedforward();
        }
    }
}

void NeuralNetwork::input( VectorD val )
{
    if ( val.size() != _layers[0].size() )
    {
        throw std::length_error( "Input vector length does not match input layer length." );
    }

    for ( int i = 0; i < val.size(); ++i )
    {
        _layers[0][i].input( val[i] );
    }
}

VectorD NeuralNetwork::get_output()
{
    VectorD tmp;
    for ( auto neuron: _layers.back() )
    {
        tmp.push_back( neuron.get_activation() );
    }

    return tmp;
}

int NeuralNetwork::get_output_int()
{
    VectorD tmp = get_output();
    return max_element( tmp.begin(), tmp.end() ) - tmp.begin();
}
