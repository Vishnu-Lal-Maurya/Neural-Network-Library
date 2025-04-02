#ifndef RELU_H
#define RELU_H

#include "ActivationFunction.hpp"
#include <algorithm>

namespace NN
{
    class ReLU final : public ActivationFunction{
    public:
        row activate(const row &input) const override {
            // std::cout << "ReLu is called for activation.\n";
            row result { input };
            for(auto &ele: result){
                ele = std::max(ele,0.0);
            }
            return result;
        };

        row derivate(const row &input) const override {
            row result{ input };
            for(auto& ele: result){
                ele = ((ele>=0.0) ? 1.0 : 0.0);
            }
            return result;
        }

        std::unique_ptr<ActivationFunction> clone() const override{
            return std::make_unique<ReLU>();
        }
        
        virtual ~ReLU() = default;
    };
};

#endif
