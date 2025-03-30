#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.hpp"
#include <cmath>

namespace NN{
    class Sigmoid final: public ActivationFunction{
    public:
        row activate(const row& input) override{
            int length { static_cast<int>(input.size()) };
            row result(length);

            for(int i{0}; i<length; ++i){
                result[i] = eval(input[i]);
            }

            return std::move(result);
        }

        row derivate(const row& input) override{
            int length { static_cast<int>(input.size()) };
            row result(length);

            for(int i{0}; i<length; ++i){
                result[i] = (eval(input[i]) * (1.0 - eval(input[i])));
            }

            return std::move(result);
        }

    private:
        double eval(double x){
            return 1.0/(1.0 + exp(-x));
        }
    };
}

#endif 
