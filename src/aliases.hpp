#ifndef ALIASES_H
#define ALIASES_H
#include <iostream>
#include <vector>

namespace NN{
    using row = std::vector<double>;
    using matrix = std::vector<std::vector<double>>;

    std::ostream& operator<<(std::ostream& out,const row& v){
        out << '[';
        std::size_t length {v.size()};
        for(std::size_t i{0}; i<length; ++i){
            out << v[i];
            if(i!=length-1) out << ", ";
        }
        return out << ']';
    }

    std::ostream& operator<<(std::ostream& out,const matrix& v){
        out << '[';
        for(std::size_t i{0}; i<v.size(); ++i){
            out << v[i];
            if(i!=v.size()-1){
                out << '\n';
            }
        }
        return out << ']';
    }

    std::size_t toUZ(int x){
        if (x < 0) throw std::invalid_argument("Negative value cannot be converted to std::size_t");
        return static_cast<std::size_t>(x);
    }
};

#endif
