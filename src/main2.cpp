#include<iostream>
#include <fstream>

int main(){
    std::ofstream fout{ "D:\\Neural-Network-Library\\Simple-Datasets\\Quadratic.csv" };
    for(int i=-5; i<=5; i++){
        for(double j=0.0; j<=0.9; j+=0.1){
            fout << (i+j) << ", " << (i+j)*(i+j) << '\n';
        }
    }
    return 0;
}