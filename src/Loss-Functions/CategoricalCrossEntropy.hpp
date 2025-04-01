#ifndef CATEGORICALCROSSENTROPY_H
#define CATEGORICALCROSSENTROPY_H

#include "LossFunction.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace NN
{
    class CategoricalCrossEntropy final : public LossFunction
    {
    public:
        double computeCost(const row &yActual, const row &yPredicted) const override {
            double result{0.0};
            for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
                result -= yActual[toUZ(i)] * (log(std::max(yPredicted[toUZ(i)],1e-320)));
            }
            return result;
        }

        row derivative(const row &yActual, const row &yPredicted) const override {
            row result{};
            result.reserve(yActual.size());
            for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
                result.push_back(yPredicted[toUZ(i)] - yActual[toUZ(i)]);
            }
            return result;
        }
    };
};

#endif
