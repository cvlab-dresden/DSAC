/*
Copyright (c) 2016, TU Dresden
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <fstream>

/** Methods for reading and writing base and complex types to binary files. 
 * Everything is heavily overloaded. You can call these methods for any 
 * supported complex type and it will recursively deconstruct it. */

namespace jp
{
    /**
     * @brief Write basic types (double, float, etc).
     * 
     * @param file Binary file to write to.
     * @param b Value to write.
     * @return void
     */
    template<class T>
    void write(std::ofstream& file, const T& b)
    {
	file.write((char*) &b, sizeof(T));
    }
    
    /**
     * @brief Read basic types (double, float, etc).
     * 
     * @param file Binary file to read  from.
     * @param b Value to read.
     * @return void
     */
    template<class T>
    void read(std::ifstream& file, T& b)
    {
	file.read(reinterpret_cast<char*>(&b), sizeof(T));
    }

    /**
     * @brief Write vectors.
     * 
     * @param file Binary file to write to.
     * @param v Vector to write.
     * @return void
     */
    template<class T>
    void write(std::ofstream& file, const std::vector<T>& v)
    {
	write<unsigned>(file, v.size());
	for(unsigned i = 0; i < v.size(); i++)
	    write(file, v[i]);
    }
    
    /**
     * @brief Read vectors.
     * 
     * @param file Binary file to read  from.
     * @param v Vector to read.
     * @return void
     */
    template<class T>
    void read(std::ifstream& file, std::vector<T>& v)
    {
	unsigned size;
	read<unsigned>(file, size);
	v.resize(size);
	for(unsigned i = 0; i < v.size(); i++)
	{
	    read(file, v[i]);
	}
    }

    /**
     * @brief Write maps.
     * 
     * @param file Binary file to write to.
     * @param m Map to write.
     * @return void
     */
    template<class T1, class T2>
    void write(std::ofstream& file, const std::map<T1, T2>& m)
    {
	write<unsigned>(file, m.size());
	for(typename std::map<T1, T2>::const_iterator it = m.begin(); it != m.end(); it++)
	{
	    write(file, it->first);
	    write(file, it->second);
	}
    }

    /**
     * @brief Read maps.
     * 
     * @param file Binary file to read  from.
     * @param m Map to read.
     * @return void
     */
    template<class T1, class T2>
    void read(std::ifstream& file, std::map<T1, T2>& m)
    {
	unsigned size;
	T1 key;
	T2 value;
	read<unsigned>(file, size);
	for(unsigned i = 0; i < size; i++)
	{
	    read(file, key);
	    read(file, value);
	    m[key] = value;
	}
    }    
    
    /**
     * @brief Write OpenCV matrices.
     * 
     * @param file Binary file to write to.
     * @param m Matrix to write.
     * @return void
     */
    template<class T>
    void write(std::ofstream& file, const cv::Mat_<T>& m)
    {
	write<int>(file, m.rows);
	write<int>(file, m.cols);      
	for(unsigned i = 0; i < m.rows; i++)
	for(unsigned j = 0; j < m.cols; j++)
	    write(file, m(i, j));
    }
    
    /**
     * @brief Read OpenCV matrices.
     * 
     * @param file Binary file to read  from.
     * @param m Matrix to read.
     * @return void
     */
    template<class T>
    void read(std::ifstream& file, cv::Mat_<T>& m)
    {
	int rows, cols;
	read<int>(file, rows);
	read<int>(file, cols);
	m = cv::Mat_<T>(rows, cols);
	for(unsigned i = 0; i < rows; i++)
	for(unsigned j = 0; j < cols; j++)
	    read(file, m(i, j));
    }
    
    /**
     * @brief Write OpenCV vectors.
     * 
     * @param file Binary file to write to.
     * @param v Vector to write.
     * @return void
     */
    template<class T, int dim>
    void write(std::ofstream& file, const cv::Vec<T, dim>& v)
    {
	for(unsigned i = 0; i < dim; i++)
	    write(file, v[i]);
    }
    
    /**
     * @brief Read OpenCV vectors.
     * 
     * @param file Binary file to read  from.
     * @param v Vector to read.
     * @return void
     */
    template<class T, int dim>
    void read(std::ifstream& file, const cv::Vec<T, dim>& v)
    {
	for(unsigned i = 0; i < dim; i++)
	    read(file, v[i]);
    }
    
    /**
     * @brief Creates a binary file from the given file name and write the given value. Close the file afterwards.
     * 
     * @param file File name of the file to create.
     * @param b Value to write.
     * @return void
     */
    template<class T>
    void write(std::string& fileName, const T& b)
    {
        std::ofstream file;
	file.open(fileName, std::ofstream::binary);  
	jp::write(file, b);
	file.close();
    }
}