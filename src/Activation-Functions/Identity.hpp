#ifndef IDENTITY_H
#define IDENTITY_H

#include "ActivationFunction.hpp"
#include <algorithm>

namespace NN
{
    class Identity final : public ActivationFunction{
    public:
        row activate(const row &input) const override {
            return input;
        };

        row derivate(const row &input) const override {
            std::size_t length { input.size() };
            row result(length,1.0);
            return result;
        }

        std::unique_ptr<ActivationFunction> clone() const override{
            return std::make_unique<Identity>();
        }
        
        int getEnumIndex() const {
            return ActivationFunctionsEnum::identity;
        }

        virtual ~Identity() = default;
    };
};

#endif
