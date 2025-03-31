#ifndef CLASSIFICATIONMETRICS_H
#define CLASSIFICATIONMETRICS_H

#include "aliases.hpp"

namespace NN{
    class Metric{
    public:
        Metric()
        :
        {

        }
    
    private:
        row yActual{}, yPredicted{}; 
    };
};

#endif