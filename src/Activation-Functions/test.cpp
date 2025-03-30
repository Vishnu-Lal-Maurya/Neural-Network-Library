#include <ReLU.hpp>

int main(){
    ReLU relu;
    MyMatrix<long double> input{2.0l,-1.0l,3.0l,5.0l,-5.0l};
    MyMatrix<long double> output = relu.activate(input);
    return 0;
}