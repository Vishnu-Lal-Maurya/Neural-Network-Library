#ifndef CLASSIFICATIONMETRICS_H
#define CLASSIFICATIONMETRICS_H

#include "aliases.hpp"
#include <cassert>

namespace NN{
    class ClassificationMetrics{
    public:
        ClassificationMetrics(const row& yActual, const row& yPredicted)
        :   m_yActual(yActual), m_yPredicted(yPredicted)
        {
            assert(yActual.size() == yPredicted.size() && "Mismatch size of yActual and yPredicted");
        }
        
        double getAccuracy(){
            int correctPredictions{0};
            double totalPredictions{static_cast<double>(m_yActual.size())};
            for(int i{0}; i < static_cast<int>(m_yActual.size()); ++i){
                correctPredictions += (m_yActual[i] == m_yPredicted[i]);
            }

            return correctPredictions/totalPredictions;
        }


    
    private:
        row m_yActual{}, m_yPredicted{}; 
    };
};

#endif