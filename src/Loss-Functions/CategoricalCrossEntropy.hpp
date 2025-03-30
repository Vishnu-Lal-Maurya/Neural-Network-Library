#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include "LossFunction.hpp"
#include <iostream>
#include <cmath>

namespace NN
{
    class CategoricalCrossEntropy final : public LossFunction
    {
    public:
        double computeCost(const row &yActual, const row &yPredicted) override {
            double result{0.0};
            for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
                result -= yActual[i] * (log(yPredicted[i]));
            }
            return result;
        }

        row derivative(const row &yActual, const row &yPredicted) override {
            row result{};

            return result;
        }
    };
};

#endif
