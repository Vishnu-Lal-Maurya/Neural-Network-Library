#ifndef IDENTITY_H
#define IDENTITY_H

#include "ActivationFunction.hpp"
#include <algorithm>

namespace NN
{
    class Identity final : public ActivationFunction{
    public:
        row activate(const row &input) const override {

            std::size_t length{ input.size() };
            row result(length);

            for(int i{0}; i<length; ++i){
                result[toUZ(i)] = input[toUZ(i)];
            }

            return result;
        };

        row derivate(const row &input) const override {

            int length { input.size() };
            row result(length);

            for(int i{0}; i<length; ++i){
                result[toUZ(i)] = 1.0;
            }

            return result;
        }
    };
};

#endif
