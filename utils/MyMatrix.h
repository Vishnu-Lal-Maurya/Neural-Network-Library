#ifndef MYMATRIX_H
#define MYMATRIX_H

#include <algorithm>
#include <vector>
#include <initializer_list>



template<typename T>
class MyMatrix
{

public:

   MyMatrix()=default;
   
   explicit MyMatrix(int x, int y)
   : m_x{x}, m_y{y}, m_matrix{std::vector<std::vector<T>>{static_cast<std::size_t>(x),std::vector<T>{static_cast<std::size_t>(x)}};
   {
   }
   
   explicit MyMatrix(std::initializer<T>)
   
   
   
   

private:
   
   int m_x{};
   int m_y{};
   std::vector<std::vector<T>> m_matrix{};
   
   
}; 
   
#endif