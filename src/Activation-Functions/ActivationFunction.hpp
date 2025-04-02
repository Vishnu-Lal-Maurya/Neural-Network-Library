#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <memory>
#include "../aliases.hpp"

namespace NN{
    class ActivationFunction
    {
    public:
        virtual row activate(const row& input) const = 0;
        virtual row derivate(const row& input) const = 0;
        virtual std::unique_ptr<ActivationFunction> clone() const = 0;
        virtual ~ActivationFunction() = default;
    };
}


#endif
