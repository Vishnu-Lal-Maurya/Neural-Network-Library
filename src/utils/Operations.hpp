#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "../aliases.hpp"
#include <cassert>
#include <chrono>
#include <random>
#include <functional>

namespace NN{


    void assertDimensions(const row& v1, const row& v2){
        assert(v1.size() == v2.size() && "Lengths Don't match");
    }

    void assertDimensions(const matrix& v1, const matrix& v2){
        assert(!v1.empty() && !v2.empty() && "Matrix must have at least one row");
        assert(v1.size() == v2.size() && "number of rows must match");
        assert(v1.size() && "number of rows should be non-zero");
        assert(v1[0].size() == v2[0].size() && "number of columns must match");
    }

    row elementWiseRowOperations(const row& v1, const row& v2, const std::function<double(double, double)>& operation){
        assertDimensions(v1, v2);

        std::size_t length{ v1.size() };
        row result(length);

        for(int i{0}; i < length; ++i){
            if (operation.target_type() == typeid(std::divides<double>)) {
                assert(v2[toUZ(i)] != 0.0 && "Division by zero not allowed");
            }
            result[toUZ(i)] = operation(v1[toUZ(i)], v2[toUZ(i)]);
        }
        
        return result;
    }


    matrix elementWiseMatrixOperations(const matrix& m1, const matrix& m2, const std::function<double(double,double)>& operation){
        assertDimensions(m1,m2);
        std::size_t numRows{m1.size()};
        matrix result(numRows);
        for(int i{0}; i < numRows; ++i){
            result[toUZ(i)] = elementWiseRowOperations(m1[toUZ(i)], m2[toUZ(i)], operation);
        }
        return result;
    }

    row add(const row& v1, const row& v2){
        return elementWiseRowOperations(v1, v2, std::plus<double>());
    }

    row sub(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::minus<double>());
    }

    row mul(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::multiplies<double>());
    }

    row div(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::divides<double>());
    }

    double dot(const row& v1, const row& v2){
        row temp { elementWiseRowOperations(v1, v2, std::multiplies<double>()) };
        return std::accumulate(temp.begin(), temp.end(), 0.0);
    }

    row mul(const row& v1, const double val){
        row v2 (v1.size(),val);
        return elementWiseRowOperations(v1,v2,std::multiplies<double>());
    }

    row div(const row& v1, const double val){
        row v2 (v1.size(),val);
        return elementWiseRowOperations(v1,v2,std::divides<double>());
    }

    row mul(const double val, const row& v1){
        return mul(v1, val);
    }

    row div(const double val, const row& v1){
        return div(v1, val);
    }

    matrix mul(const matrix& m1, const double val){
        matrix m2 (m1.size(),row(m1[0].size(),val));
        return elementWiseMatrixOperations(m1,m2,std::multiplies<double>());
    }

    matrix mul(const double val, const matrix& m1){
        return mul(m1,val);
    }


    matrix add(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2, std::plus<double>());
    }
    
    matrix sub(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::minus<double>());
    }

    matrix mul(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::multiplies<double>());
    }

    matrix div(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::divides<double>());
    }


    matrix transpose(const matrix& v){
        matrix result{};
        for(int j{0}; j < v[0].size(); ++j){
            row temp{};
            for(int i{0}; i < v.size(); ++i){
                temp.push_back(v[toUZ(i)][toUZ(j)]);
            }
            result.push_back(temp);
        }
        return result;
    }


    matrix matMul(const matrix& m1, const matrix& m2){
        assert(m1[0].size() == m2.size() && "Dimensions are not compatiable for Matrix Multiplication");

        matrix result(m1.size(), row(m2[0].size(),0));

        for(int k{0}; k < m2.size() ; ++k){
            for(int i{0}; i < m1.size(); ++i){
                double temp{m1[toUZ(i)][toUZ(k)]};
                for(int j{0}; j < m2[0].size(); ++j){
                    result[toUZ(i)][toUZ(j)] += temp * m2[toUZ(k)][toUZ(j)];
                }
            }
        }

        return result;
    }

    row matToRow(const matrix& m1){
        if(m1.size()==1) return row{m1[0]};
        assert(m1[0].size() == 1 && "number of columns should be 1");
        row result(m1.size());
        for(int k{0}; k < static_cast<int>(m1.size()) ; ++k){
            result[toUZ(k)] = m1[toUZ(k)][0];
        }
        return result;
    }

    matrix rowToColMatrix(const row& v1){
        matrix result(v1.size(),row(1,0));
        for(int k{0}; k < static_cast<int>(v1.size()) ; ++k){
            result[toUZ(k)][0] = v1[toUZ(k)];
        }
        return result;
    }


    matrix matMul(const matrix& m1, const row& v2){
        matrix m2(1,row(v2.size(),0));
        for(int k{0}; k < static_cast<int>(v2.size()) ; ++k){
            m2[0][toUZ(k)] = v2[toUZ(k)];
        }
        return matMul(m1,m2);
    }

    
    row operator+(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::plus<double>());
    }

    row operator-(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::minus<double>());
    }

    row operator*(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::multiplies<double>());
    }

    row operator/(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,std::divides<double>());
    }

    matrix operator+(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::plus<double>());
    }
    
    matrix operator-(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::minus<double>());
    }

    matrix operator*(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::multiplies<double>());
    }

    matrix operator/(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,std::divides<double>());
    }

}

#endif