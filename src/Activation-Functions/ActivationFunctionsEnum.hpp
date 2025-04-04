#ifndef ACTIVATIONFUNCTIONSENUM_H
#define ACTIVATIONFUNCTIONSENUM_H

#include <memory>
#include "ActivationFunction.hpp"
#include "Argmax.hpp"
#include "Identity.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Softmax.hpp"
#include "Tanh.hpp"

namespace NN{

    std::unique_ptr<ActivationFunction> getActivationFunctionFromEnum(int enumNumber){
        switch (enumNumber)
        {
        case argmax: return std::make_unique<Argmax>();
        case identity: return std::make_unique<Identity>();
        case relu: return std::make_unique<ReLU>();
        case sigmoid: return std::make_unique<Sigmoid>();
        case softmax: return std::make_unique<Softmax>();
        case tanh: return std::make_unique<Tanh>();
        default:
            throw "not a valid enum\n";
        }
    }

}

#endif