#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.hpp"
#include <cmath>

namespace NN{
    class Sigmoid final: public ActivationFunction{
    public:
        row activate(const row& input) const override{
            // std::cout << "I'm actually called\n";
            row result { input };

            for(auto& ele: result){
                ele = eval(ele);
            }

            return result;
        }

        row derivate(const row& input) const override{
            row result(input);

            for(auto& ele: result){
                ele = (eval(ele) * (1.0 - eval(ele)));
            }

            return result;
        }

        virtual ~Sigmoid() = default;

    private:
        double eval(double x) const {
            return 1.0/(1.0 + exp(-x));
        }
    };
}

#endif 
