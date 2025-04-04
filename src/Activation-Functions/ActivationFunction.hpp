#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <memory>
#include "../aliases.hpp"

namespace NN{

    enum ActivationFunctionsEnum{
        argmax,
        identity,
        relu,
        sigmoid,
        softmax,
        tanh,
        numberOfActivationsFunctions,
    };

    class ActivationFunction
    {
    public:
        virtual row activate(const row& input) const = 0;
        virtual row derivate(const row& input) const = 0;
        virtual std::unique_ptr<ActivationFunction> clone() const = 0;
        // below function will be used for encoding and decoding of Activation Function
        virtual int getEnumIndex() const = 0;
        virtual ~ActivationFunction() = default;
    };
}


#endif
