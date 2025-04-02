#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

#include "LossFunction.hpp"
#include <algorithm>

namespace NN{

    class MeanSquaredError final: public LossFunction{
    public:

        double computeCost(const row& yActual, const row& yPredicted) const override {

            assert(!yActual.empty() && "Input vectors must not be empty.");
            assert(yActual.size()==yPredicted.size() && "yActual and yPredicted should have same size");
            row temp { mul(sub(yActual, yPredicted), sub(yActual, yPredicted)) };
            return std::accumulate(temp.begin(), temp.end(), 0.0) / (2.0 * static_cast<double>(yActual.size()));
        
        }

        row derivative(const row& yActual, const row& yPredicted) const override {

            assert(!yActual.empty() && "Input vectors must not be empty.");
            assert(yActual.size()==yPredicted.size() && "yActual and yPredicted should have same size");
            return div(sub(yPredicted,yActual),static_cast<double>(yActual.size()));

        }
    };

};

#endif