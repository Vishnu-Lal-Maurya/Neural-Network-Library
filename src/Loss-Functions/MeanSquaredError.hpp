#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

#include "LossFunction.hpp"
#include <algorithm>

namespace NN{

    class MeanSquaredError final: public LossFunction{
    public:
        double computeCost(const row& yActual, const row& yPredicted) override {

            double result {0.0};
            int length = yActual.size();

            for(int i{0}, i<length; ++i){
                result += (yActual[i] - yPredicted[i]) * (yActual[i] - yPredicted[i]);
            }
            result /= length;
            
            return result;
        }

        double derivative(const row& yActual, const row& yPredicted) override {

            double result {0.0};

            return result;
        }
    };

};

#endif