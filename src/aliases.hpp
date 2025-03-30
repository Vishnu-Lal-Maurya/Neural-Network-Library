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
};

#endif
