#ifndef REGRESSIONMETRICS_H
#define REGRESSIONMETRICS_H

#include "../aliases.hpp"
#include <cassert>

namespace NN{

    double calculateMeanAbsoluteError(const row& yPredicted, const row& yActual) {
        assertDimensions(yPredicted, yActual);
        double result{0.0};

        for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
            result += abs(yActual[toUZ(i)]-yPredicted[toUZ(i)]);
        }

        return result / (yActual.size());
    }

    double calculateMeanSquaredError(const row& yPredicted, const row& yActual) {
        assertDimensions(yPredicted, yActual);
        double result{0.0};

        for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
            double difference{yActual[toUZ(i)]-yPredicted[toUZ(i)]};
            result += difference * difference;
        }

        return result / (yActual.size());
    }
}


#endif