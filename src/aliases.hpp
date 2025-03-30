#ifndef ALIASES_H
#define ALIASES_H
#include <iostream>
#include <vector>

namespace NN{
    using row = std::vector<double>;
    using matrix = std::vector<std::vector<double>>;

    std::ostream& operator<<(std::ostream& out, row& v){
        out << '[';
        int length { static_cast<int>(v.size()) };
        for(int i=0; i<length; ++i){
            out << v[i];
            if(i!=length-1) out << ", ";
        }
        return out << ']';
    }
};

#endif
