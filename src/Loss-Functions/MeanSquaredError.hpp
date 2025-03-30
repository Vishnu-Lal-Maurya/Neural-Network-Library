#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

#include "LossFunction.hpp"
#include <algorithm>

namespace NN{

    class MeanSquaredError final: public LossFunction{
    public:
        double computeCost(const row& yActual, const row& yPredicted) override {

            double result {0.0};
            int length = static_cast<int>(yActual.size());

            for(int i{0}; i<length; ++i){
                std::size_t x { static_cast<std::size_t>(i) };
                result += (yActual[x] - yPredicted[x]) * (yActual[x] - yPredicted[x]);
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