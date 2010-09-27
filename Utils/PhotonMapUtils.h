// Utils functions 'n stuff for photon mapping
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _PHOTON_MAP_UTILS_H_
#define _PHOTON_MAP_UTILS_H_

#include <sstream>
#include <string>

inline std::string ToBitmap(int bmp){
    std::ostringstream out;
    if (bmp & 1)
        out << "[1";
    else
        out << "[0";

    for (unsigned int i = 1; i < 8 * sizeof(int); ++i){
        if (bmp & 1<<i)
            out << ", 1";
        else
            out << ", 0";
    }
    out << "]";

    return out.str();
}

#endif
