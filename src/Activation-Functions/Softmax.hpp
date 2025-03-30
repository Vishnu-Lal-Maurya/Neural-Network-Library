#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "ActivationFunction.hpp"
#include <cmath>

namespace NN{
    class Softmax final: public ActivationFunction
    {
    public:
        row activate(const row& input) override {
            row result{};
            double denominator{};
            for(auto i : input){
                denominator += exp(i);
            }
            
            for(auto i : input){
                result.push_back(exp(i) / denominator);
            }
            return result;
        };

        row derivate(const row& input) override{
            row result{activate(input)};
            for(auto &i : result){
                i *= (1 - i);
            }
            return result;
        }
    };
};


#endif
