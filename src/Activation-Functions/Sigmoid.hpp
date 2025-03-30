#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.hpp"

class Sigmoid: public ActivationFunction{
public:
    MyMatrix<long double> activate(MyMatrix<long double>& input){
        int rowSize { input.getCols() };
        MyMatrix<long double> result { rowSize };
    } 

    MyMatrix<long double> derivative(MyMatrix<long double>& input);
private:

};


#endif SIGMOID_H
