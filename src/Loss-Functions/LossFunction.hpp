#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H
#include "../aliases.hpp"


namespace NN{
   class LossFunction
   {
   public:
       virtual double computeCost(const row& yActual, const row& yPredicted) const = 0;
       virtual row derivative(const row& yActual, const row& yPredicted) const = 0;
       virtual ~LossFunction() = default;
   };
}

#endif