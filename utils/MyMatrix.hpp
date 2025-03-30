#ifndef MYMATRIX_H
#define MYMATRIX_H

#include <algorithm>
#include <vector>
#include <initializer_list>
#include <iostream>



template<typename T>
class MyMatrix
{

public:

   MyMatrix()=default;

   // checker in constructor for x>0 and y>0

   explicit MyMatrix(int cols)
   : MyMatrix(1,cols)
   {
   }
   
   explicit MyMatrix(int rows, int cols)
   : m_rows{rows}, m_cols{cols},
   m_matrix{std::vector<std::vector<T>>(m_rows,std::vector<T>(m_cols))}
   {
   }
   
   explicit MyMatrix(std::initializer_list<T> list)
   : MyMatrix(1,static_cast<int>(list.size()))
   {
      std::copy(list.begin(),list.end(),m_matrix[0]);
   }
   
   explicit MyMatrix(std::initializer_list<std::initializer_list<T>> list)
   : m_rows{static_cast<int>(list.size())}, m_cols{static_cast<int>(list.begin().size())}
   {
      std::copy(list.begin(),list.end(),m_matrix);
   }

   MyMatrix(MyMatrix& other) = default;

   MyMatrix& operator=(MyMatrix& other) = default;

   friend std::ostream& operator<<(std::ostream& out, const MyMatrix& matrix){
      out << "[ ";
      for(auto &row: matrix.m_matrix){
         out << "[ ";
         for(auto &cols:row){
            out << cols << ", ";
         }
         out << "]";
      }
      out << " ]";
      return out;
   }



   



   


   int getRows(){return m_rows;}

   int getCols(){return m_cols;}

   
   
   

private:
   
   int m_rows{};
   int m_cols{};
   std::vector<std::vector<T>> m_matrix{};
   
   
}; 
   
#endif