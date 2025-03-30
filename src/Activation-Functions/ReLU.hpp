#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.hpp"
#include <algorithm>

namespace NN
{
    class ReLU final : public ActivationFunction{
    public:
        row activate(const row &input) const override {

            int length{ static_cast<int> (input.size()) };
            row result(length);

            for(int i{0}; i<length; ++i){
                result[i] = std::max(input[i], 0.0);
            }

            return result;
        };

        row derivate(const row &input) const override {

            int length { static_cast<int> (input.size()) };
            row result(length);

            for(int i{0}; i<length; ++i){
                result[i] = (input[i] > 0);
            }

            return result;
        }
    };
};

#endif
