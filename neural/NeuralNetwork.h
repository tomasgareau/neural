#ifndef __NEURALNETWORK_H
#define __NEURALNETWORK_H

#include <vector>
#include <iostream>
#include <time.h>
#include "Neuron.h"
#include "Data.h"

using std::vector;

class NeuralNetwork 
{
    private:
        vector<vector<Neuron*>> _layers;

    public:
        NeuralNetwork( vector<int>& sizes );
        void init();
        void stochastic_gradient_descent( Data& training_data, int epochs, int mini_batch_size, double eta, Data* test_data );
        void update_mini_batch( DataIterator start, DataIterator stop, double eta );
        void input( VectorD val );
        void feedforward();
        VectorD get_output();
        long get_output_int();
        void clean_up();
};

#endif
