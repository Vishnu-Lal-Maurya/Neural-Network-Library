#ifndef CLASSIFICATIONMETRICS_H
#define CLASSIFICATIONMETRICS_H

#include "../aliases.hpp"
#include <cassert>

namespace NN{
    double getAccuracy(const row& yPredicted, const row& yActual) {
        assertDimensions(yPredicted, yActual);
        int correctPredictions{0};
        double totalPredictions{static_cast<double>(yActual.size())};
        for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
            correctPredictions += (yActual[toUZ(i)] == yPredicted[toUZ(i)]);
        }

        return correctPredictions/totalPredictions;
    }

    double getPrecision(const row& yPredicted, const row& yActual, int trueClass) {
        int truePositives{}, falsePositives{};
        for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
            if(yPredicted[toUZ(i)] == trueClass){
                if(yActual[toUZ(i)] == trueClass){
                    truePositives++;
                }
                else{
                    falsePositives++;
                }
            }
        }
        return static_cast<double>(truePositives) / (truePositives + falsePositives);
    }

    double getAvgPrecision(const row& yPredicted, const row& yActual, int totalClasses) {
        double result{};
        for(int i{0}; i< totalClasses; ++i){
            result += getPrecision(yPredicted, yActual, i);
        }
        return result / totalClasses;
    }
    

    double getRecall(const row& yPredicted, const row& yActual, int trueClass) {
        int truePositives{}, falseNegatives{};
        for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
            if(yActual[i] == trueClass){
                if(yPredicted[i] == trueClass){
                    truePositives++;
                }
                else{
                    falseNegatives++;
                }
            }
        }
        return static_cast<double>(truePositives) / (truePositives + falseNegatives);
    }

    double getAvgRecall(const row& yPredicted, const row& yActual, int totalClasses) {
        double result{};
        for(int i{0}; i< totalClasses; ++i){
            result += getRecall(yPredicted, yActual, i);
        }
        return result / totalClasses;
    }

    double getF1Score(const row& yPredicted, const row& yActual, int totalClasses) {
        double avgPrecision(getAvgPrecision(yPredicted, yActual, totalClasses));
        double avgRecall(getAvgRecall(yPredicted, yActual, totalClasses));
        return (2 * avgPrecision * avgRecall) / (avgPrecision + avgRecall);
    }
};

#endif