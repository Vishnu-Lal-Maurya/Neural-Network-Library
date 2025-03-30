#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../aliases.hpp"

namespace NN{
    class ActivationFunction
    {
    public:
        virtual row activate(row& input) = 0;
        virtual row derivate(row& input) = 0;
    };
}


#endif
