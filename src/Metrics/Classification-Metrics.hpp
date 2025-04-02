#ifndef CLASSIFICATIONMETRICS_H
#define CLASSIFICATIONMETRICS_H

#include "../aliases.hpp"
#include <cassert>

namespace NN{
    double calculateAccuracy(const row& yPredicted, const row& yActual) {
        assertDimensions(yPredicted, yActual);
        int correctPredictions{0};
        double totalPredictions{static_cast<double>(yActual.size())};
        for(int i{0}; i < static_cast<int>(yActual.size()); ++i){
            correctPredictions += (yActual[toUZ(i)] == yPredicted[toUZ(i)]);
        }

        return correctPredictions/totalPredictions;
    }

    double calculatePrecision(const row& yPredicted, const row& yActual, int trueClass) {
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

    double calculateAvgPrecision(const row& yPredicted, const row& yActual, int totalClasses) {
        double result{};
        for(int i{0}; i< totalClasses; ++i){
            result += calculatePrecision(yPredicted, yActual, i);
        }
        return result / totalClasses;
    }
    

    double calculateRecall(const row& yPredicted, const row& yActual, int trueClass) {
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

    double calculateAvgRecall(const row& yPredicted, const row& yActual, int totalClasses) {
        double result{};
        for(int i{0}; i< totalClasses; ++i){
            result += calculateRecall(yPredicted, yActual, i);
        }
        return result / totalClasses;
    }

    double calculateF1Score(const row& yPredicted, const row& yActual, int totalClasses) {
        double avgPrecision{ calculateAvgPrecision(yPredicted, yActual, totalClasses) };
        double avgRecall{ calculateAvgRecall(yPredicted, yActual, totalClasses) };
        return (2 * avgPrecision * avgRecall) / (avgPrecision + avgRecall);
    }
};

#endif