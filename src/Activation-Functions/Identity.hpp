#ifndef IDENTITY_H
#define IDENTITY_H

#include "ActivationFunction.hpp"
#include <algorithm>

namespace NN
{
    class Identity final : public ActivationFunction{
    public:
        row activate(const row &input) const override {
            return std::move_if_noexcept(input);
        };

        row derivate(const row &input) const override {
            int length { input.size() };
            row result(length,1.0);
            return std::move_if_noexcept(result);
        }
    };
};

#endif
