#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H
#include "../aliases.hpp"


namespace NN{
   class LossFunction
   {
   public:
       virtual row computeCost(const row& input) = 0;
       virtual row derivative(const row& input) = 0;
   };
}

#endif