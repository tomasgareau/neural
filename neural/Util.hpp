//
//  Util.hpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#ifndef Util_hpp
#define Util_hpp

#include <iostream>
#include <random>

#if DEBUG
    #define DEBUG_PRINT(x) do { std::cout << x << std::flush; } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

namespace Util
{
    typedef std::pair<std::vector<double>, std::vector<double>> DataPair;
    enum LinkDirection
    {
        PREV,
        NEXT
    };
    
    static std::random_device rd;
    static std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    static std::mt19937 generator(seed);
    static std::normal_distribution<double> distribution( 0.0, 1.0 );
    
    double sigmoid( double val );
    double sigmoid_prime( double val );
    
    struct Link
    {
        double weight;
        double output;
        double error;
        
        Link()
        {
            weight = Util::distribution( Util::generator );
            output = 0.0;
            error = 0.0;
        }
    };
}

#endif /* Util_hpp */
