#ifndef ARGMAX_H
#define ARGMAX_H

#include "ActivationFunction.hpp"
#include <cmath>
#include <algorithm>

namespace NN{   
    class Argmax final: public ActivationFunction
    {
    public:
        row activate(const row& input) const override {
            row result(input.size(), 0.0);
            int maxIndex{ static_cast<int>(max_element(input.begin(), input.end()) - input.begin()) };
            result[toUZ(maxIndex)] = 1.0;
            return result;
        };

        row derivate(const row& input) const override{
            throw "Can't take derivative of ArgMax";
            return row{};
        }

        std::unique_ptr<ActivationFunction> clone() const override{
            return std::make_unique<Argmax>();
        }

        int getEnumIndex() const {
            return ActivationFunctionsEnum::argmax;
        }
        
        virtual ~Argmax() = default;
    };
};


#endif
