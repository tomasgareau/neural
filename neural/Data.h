#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <tuple>
#include "Types.h"

static std::random_device _r;
static std::seed_seq _seed{_r(), _r(), _r(), _r(), _r(), _r(), _r(), _r()};
static std::mt19937 _generator( _seed );

class Data
{
    private:
        DataVector _data;

    public:

        void add_data( VectorD input, VectorD output )
        {
            auto data_tuple = std::make_tuple( input, output );
            _data.push_back( data_tuple );
        }

        int size()
        {
            return _data.size();
        }

        void shuffle()
        {
            std::shuffle( _data.begin(), _data.end(), _generator );
        }

        DataIterator begin()
        {
            return _data.begin();
        }

        DataIterator end()
        {
            return _data.end();
        }
};

#endif
