#ifndef CLASSIFICATIONMETRICS_H
#define CLASSIFICATIONMETRICS_H

#include "../aliases.hpp"
#include <cassert>

namespace NN{
    class ClassificationMetrics{
    public:
        ClassificationMetrics(const row& yActual, const row& yPredicted)
        :   m_yActual(yActual), m_yPredicted(yPredicted)
        {
            assert(yActual.size() == yPredicted.size() && "Mismatch size of yActual and yPredicted");
        }
        
        double getAccuracy() const {
            int correctPredictions{0};
            double totalPredictions{static_cast<double>(m_yActual.size())};
            for(int i{0}; i < static_cast<int>(m_yActual.size()); ++i){
                correctPredictions += (m_yActual[i] == m_yPredicted[i]);
            }

            return correctPredictions/totalPredictions;
        }

        double getPrecision(int trueClass) const {
            int truePositives{}, falsePositives{};
            for(int i{0}; i < static_cast<int>(m_yActual.size()); ++i){
                if(m_yPredicted[i] == trueClass){
                    if(m_yActual[i] == trueClass){
                        truePositives++;
                    }
                    else{
                        falsePositives++;
                    }
                }
            }
            return static_cast<double>(truePositives) / (truePositives + falsePositives);
        }

        double getAvgPrecision(int totalClasses) const {
            double result{};
            for(int i{0}; i< totalClasses; ++i){
                result += getPrecision(i);
            }
            return result / totalClasses;
        }
        

        double getRecall(int trueClass) const {
            int truePositives{}, falseNegatives{};
            for(int i{0}; i < static_cast<int>(m_yActual.size()); ++i){
                if(m_yActual[i] == trueClass){
                    if(m_yPredicted[i] == trueClass){
                        truePositives++;
                    }
                    else{
                        falseNegatives++;
                    }
                }
            }
            return static_cast<double>(truePositives) / (truePositives + falseNegatives);
        }

        double getAvgRecall(int totalClasses) const {
            double result{};
            for(int i{0}; i< totalClasses; ++i){
                result += getRecall(i);
            }
            return result / totalClasses;
        }

        double getF1Score(int totalClasses) const {
            double avgPrecision(getAvgPrecision(totalClasses));
            double avgRecall(getAvgRecall(totalClasses));
            return (2 * avgPrecision * avgRecall) / (avgPrecision + avgRecall);
        }
        
        
        

    private:
        row m_yActual{}, m_yPredicted{}; 
    };
};

#endif