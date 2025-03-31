#ifndef REGRESSIONMETRICS_H
#define REGRESSIONMETRICS_H

#include <aliases.hpp>
#include <cassert>

namespace NN{
    class RegressionMetrics{
    public:
        RegressionMetrics(const row& yActual, const row& yPredicted)
        : m_yActual{yActual}, m_yPredicted{yPredicted}
        {
            assert(m_yActual.size() == m_yPredicted.size() && "Mismatch size of yActual and yPredicted");
        }

        double getMAE(){
            double result{0.0};
            for(int i{0}; i < static_cast<int>(m_yActual.size()); ++i){
                result += abs(m_yActual[i]-m_yPredicted[i]);
            }
            return result / (m_yActual.size());
        }

        double getMSE(){
            double result{0.0};
            for(int i{0}; i < static_cast<int>(m_yActual.size()); ++i){
                double difference{m_yActual[i]-m_yPredicted[i]};
                result += difference * difference;
            }
            return result / (m_yActual.size());
        }

        
    private:
        row m_yActual{}, m_yPredicted{};
    };
}


#endif