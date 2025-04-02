#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "ActivationFunction.hpp"
#include <cmath>

namespace NN{
    class Softmax final: public ActivationFunction
    {
    public:
        row activate(const row& input) const override {
            std::size_t length { input.size() };
            row result{};
            result.reserve(length);

            double denominator{};
            
            for(auto i : input){
                denominator += exp(i);
            }
            
            for(auto i : input){
                result.emplace_back(exp(i) / denominator);
            }

            return result;
        };

        row derivate(const row& input) const override{
            row result{activate(input)};
            for(auto &i : result){
                i *= (1 - i);
            }
            return result;
        }

        std::unique_ptr<ActivationFunction> clone() const override{
            return std::make_unique<Softmax>();
        }
        
        virtual ~Softmax() = default;
    };
};


#endif
