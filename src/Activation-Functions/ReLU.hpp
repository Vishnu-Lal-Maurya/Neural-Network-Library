#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.hpp"

namespace NN{
    class ReLU final: public ActivationFunction
    {
    public:
        NN::row activate(NN::row& input){
            // int length = input.size();
            // NN::row result(length, 0);
        };
    };
};


#endif
