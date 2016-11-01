#ifndef __NEURON_H
#define __NEURON_H

#include <random>
#include <vector>

using std::vector;

static std::random_device r;
static std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
static std::mt19937 generator(seed);
static std::normal_distribution<double> distribution( 0.0, 1.0 );

class Neuron
{
    private:
        double _bias;
        double _activation;
        double _error;
        vector<Neuron*>* _next_layer;
        vector<double> _weights;

    public:
        Neuron();
        void init( vector<Neuron*>* next_layer );

        double get_activation();
        double get_error();
        void set_error( double val );

        void input( double val );
        void feedforward();
        void backprop();
        void gradient_descent( double eta, long mini_batch_size );

        inline static double sigmoid( double val )
        {
            return 1.0 / ( 1.0 + std::exp( -val ) );
        }

        inline static double sigmoid_prime( double val )
        {
            return sigmoid( val ) * ( 1 - sigmoid( val ) );
        }
};

#endif
