#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "../aliases.hpp"
#include <cassert>

namespace NN{

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
            for(int i{0}; i < static_cast<int>(v[0].size()); ++j){
                temp.push_back(v[i][j]);
            }
            result.push_back(temp);
        }
        return result;
    }

    matrix matMul(const matrix& v1, const matrix& v2){
        // matrix result{};
    }
}

#endif