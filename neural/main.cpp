#include <iostream>
#include <fstream>
#include <string>
#include "NeuralNetwork.h"
#include "Data.h"
#include "Types.h"

using std::string;

int read_int( std::ifstream& ifstr )
{
    uint8_t tmp[4];
    ifstr.read( (char *)&tmp[0], 4 );
    return tmp[3] | tmp[2] << 8 | tmp[1] << 16 | tmp[0] << 24;
}

VectorD read_data( std::ifstream& ifstr, size_t num )
{
    VectorUI tmp( num );
    ifstr.read( (char *)&tmp[0], num );
    VectorD ret;
    for ( int i = 0; i < num; ++i )
    {
        ret.push_back( (double) tmp[i] );
    }
    return ret;
}

VectorD read_label( std::ifstream& ifstr )
{
    VectorD ret( 10, 0.0 );
    uint8_t num[1];
    ifstr.read( (char *)&num[0], 1 );
    ret[ num[0] ] = 1.0;
    return ret;
}

int main()
{
    string directory = "/Users/tomas/coding/cpp/neural/mnist_data/";
    string training_images = directory + "train-images.idx3-ubyte";
    string training_labels = directory + "train-labels.idx1-ubyte";
    string test_images_filename = directory + "t10k-images.idx3-ubyte";
    string test_labels_filename = directory + "t10k-labels.idx1-ubyte";

    std::cout << "Reading data files... " << std::flush;

    std::ifstream train_images( training_images, std::ios::in | std::ios::binary );
    if ( !train_images.is_open() )
    {
        std::cout << "Could not open file '" << training_images << "':\n" << std::strerror( errno ) << std::endl;
        std::exit(1);
    }

    std::ifstream train_labels( training_labels, std::ios::in | std::ios::binary );
    if ( !train_labels.is_open() )
    {
        std::cout << "Could not open file '" << training_labels << "':\n" << std::strerror( errno ) << std::endl;
        std::exit(1);
    }

    Data training_data;
    int magic_number = read_int( train_images );
    int num_images = read_int( train_images );
    int num_rows = read_int( train_images );
    int num_cols = read_int( train_images );
    int img_size = num_rows * num_cols;
    int magic_number_labels = read_int( train_labels );
    int num_labels = read_int( train_labels );

    for ( int i = 0; i < num_images; ++i )
    {
        training_data.add_data( read_data( train_images, img_size ), read_label( train_labels ) );
    }

    train_images.close();
    train_labels.close();

    std::ifstream test_images( test_images_filename, std::ios::in | std::ios::binary );
    if ( !test_images.is_open() )
    {
        std::cout << "Could not open file '" << test_images_filename << "':\n" << std::strerror( errno ) << std::endl;
        std::exit(1);
    }

    std::ifstream test_labels( test_labels_filename, std::ios::in | std::ios::binary );
    if ( !test_labels.is_open() )
    {
        std::cout << "Could not open file '" << test_labels_filename << "':\n" << std::strerror( errno ) << std::endl;
        std::exit(1);
    }

    Data test_data;
    int test_magic_number = read_int( test_images );
    int test_num_images = read_int( test_images );
    int test_num_rows = read_int( test_images );
    int test_num_cols = read_int( test_images );
    int test_img_size = test_num_rows * test_num_cols;
    int test_magic_number_labels = read_int( test_labels );
    int test_num_labels = read_int( test_labels );

    for ( int i = 0; i < test_num_images; ++i )
    {
        test_data.add_data( read_data( test_images, test_img_size ), read_label( test_labels ) );
    }

    std::cout << "Done.\n" << std::flush;

    std::cout << "Creating neural network... " << std::flush;
    std::vector<int> tmp = { 784, 30, 10 };
    NeuralNetwork net( tmp );
    std::cout << "Initializing... " << std::flush;
    net.init();
    std::cout << "Done.\n" << std::flush;

    std::cout << "Starting SGD.\n" << std::flush;
    net.stochastic_gradient_descent( training_data, 30, 100, 10, &test_data );
    std::cout << "Done!" << std::endl;
}

