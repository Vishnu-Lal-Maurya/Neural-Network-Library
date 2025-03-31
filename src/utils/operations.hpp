#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "../aliases.hpp"
#include <cassert>
#include <chrono>
#include <random>

namespace NN{

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    double randInRange(double lowerBound, double upperBound){
        return std::uniform_real_distribution<double> (lowerBound, upperBound)(rng);
    }

    void assertDimensions(const row& v1, const row& v2){
        std::size_t length1{ v1.size() };
        std::size_t length2{ v2.size() };
        assert(length1 == length2 && "Lengths Don't match");
    }

    void assertDimensions(const matrix& v1, const matrix& v2){
        std::size_t rows1{ v1.size() };
        std::size_t rows2{ v2.size() };
        assert(rows1 == rows2 && "Dimensions must match");
        assert(rows1 && "Dimension should be non-zero");

        std::size_t col1{ v1[0].size() };
        std::size_t col2{ v2[0].size() };
        assert(rows1 == rows2 && "Dimensions must match");
    }

    row elementWiseRowOperations(const row& v1, const row& v2, char op){
        assertDimensions(v1, v2);

        std::size_t length{ v1.size() };
        row result(length);

        for(int i{0}; i < length; ++i){
            switch (op)
            {
            case '+':
                result[toUZ(i)] = v1[toUZ(i)] + v2[toUZ(i)];
                break;
            case '-':
                result[toUZ(i)] = v1[toUZ(i)] - v2[toUZ(i)];
                break;
            case '*':
                result[toUZ(i)] = v1[toUZ(i)] * v2[toUZ(i)];
                break;
            case '/':
                assert(v2[toUZ(i)] && "divide by zero");
                result[toUZ(i)] = v1[toUZ(i)] / v2[toUZ(i)];
                break;
            }
        }
        return std::move_if_noexcept(result);
    }

    matrix elementWiseMatrixOperations(const matrix& v1, const matrix& v2, char op){
        assertDimensions(v1,v2);
        std::size_t numRows{ v1.size() };
        std::size_t numCols{ v1[0].size() };

        matrix result(numRows, row(numCols));
        for(int i{0}; i < numRows; ++i){
            for(int j{0}; j < numCols; ++j){
                int val{};
                switch (op)
                {
                case '+':
                    val = v1[toUZ(i)][toUZ(j)] + v2[toUZ(i)][toUZ(j)];
                    break;
                case '-':
                    val = v1[toUZ(i)][toUZ(j)] - v2[toUZ(i)][toUZ(j)];
                    break;
                case '*':
                    val = v1[toUZ(i)][toUZ(j)] * v2[toUZ(i)][toUZ(j)];
                    break; 
                case '/':
                    assert(v2[toUZ(i)][toUZ(j)] && "divide by zero");
                    val = v1[toUZ(i)][toUZ(j)] / v2[toUZ(i)][toUZ(j)];
                    break; 
                }
                result[toUZ(i)][toUZ(j)] = val;
            }
        }

        return std::move_if_noexcept(result);
    }

    row add(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'+');
    }

    row sub(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'-');
    }

    row mul(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'*');
    }

    row div(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'/');
    }

    double dot(const row& v1, const row& v2){
        double result{0};
        row temp { elementWiseRowOperations(v1, v2, '*') };
        for(auto x: temp){
            result += x;
        }
        return result;
    }

    row mul(const row& v1, const double val){
        row v2 (v1.size(),val);
        return elementWiseRowOperations(v1,v2,'*');
    }
    row mul(const double val, const row& v1){
        row v2 (v1.size(),val);
        return elementWiseRowOperations(v1,v2,'*');
    }

    matrix mul(const matrix& m1, const double val){
        matrix m2 (m1.size(),row(m1[0].size(),val));
        return elementWiseMatrixOperations(m1,m2,'*');
    }
    matrix mul(const double val, const matrix& m1){
        matrix m2 (m1.size(),row(m1[0].size(),val));
        return elementWiseMatrixOperations(m1,m2,'*');
    }


    matrix add(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'+');
    }
    
    matrix sub(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'-');
    }

    matrix mul(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'*');
    }

    matrix div(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'/');
    }

    matrix transpose(const matrix& v){
        matrix result{};
        for(int j{0}; j < static_cast<int>(v[0].size()); ++j){
            row temp{};
            for(int i{0}; i < static_cast<int>(v.size()); ++i){
                temp.push_back(v[i][j]);
            }
            result.push_back(temp);
        }
        return result;
    }

    matrix matMul(const matrix& v1, const matrix& v2){
        assert(v1[0].size() == v2.size());
        matrix result(v1.size(), row(v2[0].size(),0));
        for(int k{0}; k < static_cast<int>(v2.size()) ; ++k){
            for(int i{0}; i < static_cast<int>(v1.size()); ++i){
                double temp{v1[i][k]};
                for(int j{0}; j < static_cast<int>(v2[0].size()); ++j){
                    result[i][j] += temp * v2[k][j];
                }
            }
        }
        return result;
    }

    row matToRow(const matrix& m1){
        if(m1.size()==1)return m1[0];
        assert(m1[0].size() == 1);
        row result(m1.size());
        for(int k{0}; k < static_cast<int>(m1[0].size()) ; ++k){
            result[toUZ(k)] = m1[toUZ(k)][0];
        }
        return std::move_if_noexcept(result);
    }

    matrix rowToColMatrix(const row& v1){
        matrix result(v1.size(),row(1,0));
        for(int k{0}; k < static_cast<int>(v1.size()) ; ++k){
            result[toUZ(k)][0] = v1[toUZ(k)];
        }
        return result;
    }


    matrix matMul(const row& v1, const row& v2){

        matrix m1(v1.size(),row(1,0));
        matrix m2(1,row(v2.size(),0));

        for(int k{0}; k < static_cast<int>(v2.size()) ; ++k){
            m2[0][toUZ(k)] = v2[toUZ(k)];
        }

        for(int k{0}; k < static_cast<int>(v1.size()) ; ++k){
            m1[toUZ(k)][0] = v1[toUZ(k)];
        }

        return matMul(m1,m2);

    }
    
    row operator+(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'+');
    }

    row operator-(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'-');
    }

    row operator*(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'*');
    }

    row operator/(const row& v1, const row& v2){
        return elementWiseRowOperations(v1,v2,'/');
    }

    matrix operator+(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'+');
    }
    
    matrix operator-(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'-');
    }

    matrix operator*(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'*');
    }

    matrix operator/(const matrix& v1, const matrix& v2){
        return elementWiseMatrixOperations(v1,v2,'/');
    }

    matrix randMatrix(int rows, int cols){
        matrix result(rows,row(cols));
        for(int i{0}; i < rows; ++i){
            for(int j{0}; j < cols; ++j){
                result[i][j] = randInRange(-1.0,1.0);
            }
        }
        return std::move_if_noexcept(result);
    }

    row randRow(int cols){
        row result(cols);
        for(auto& i : result){
            i = randInRange(-1.0,1.0);
        }
        return std::move_if_noexcept(result);
    }
}

#endif