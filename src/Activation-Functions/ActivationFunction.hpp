#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "MyMatrix.hpp"

// doubt: kya aisi class jisme koii member nhi hai use class bnana hota hai.

class ActivationFunction{
public:
    virtual MyMatrix<long double> activate(MyMatrix<long double>& input) = 0; 

    virtual MyMatrix<long double> derivative(MyMatrix<long double>& input) = 0;
};

#endif
