//
//  Util.cpp
//  neural
//
//  Created by Tomas Gareau on 2016-11-01.
//  Copyright © 2016 Tomas Gareau. All rights reserved.
//

#include "Util.hpp"

namespace Util
{
    double sigmoid( double val )
    {
        return 1.0 / ( 1.0 + std::exp( -val ) );
    }
    
    double sigmoid_prime( double val )
    {
        return sigmoid( val ) * ( 1 - sigmoid( val ) );
    }
}
