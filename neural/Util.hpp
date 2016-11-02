//
//  Util.hpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#ifndef Util_hpp
#define Util_hpp

#include <random>

namespace Util
{
    enum LinkDirection
    {
        PREV,
        NEXT
    };
    
    std::random_device rd;
    std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution( 0.0, 1.0 );
    
    double sigmoid( double val );
    double sigmoid_prime( double val );
    
    struct Link
    {
        double weight;
        double output;
        
        Link()
        {
            weight = Util::distribution( Util::generator );
            output = 0.0;
        }
    };
}

#endif /* Util_hpp */
