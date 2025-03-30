#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../aliases.hpp"

class ActivationFunction
{
public:
    virtual NN::row activate(NN::row& input) = 0;
    virtual NN::row derivate(NN::row& input) = 0;
};


#endif
