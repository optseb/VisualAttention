#pragma once

#include <string>
#include <vector>
#include "conn.h"
#include <morph/HdfData.h>

// Extended HdfData class with save/load std::vector<morph::nn::conn<float>> weight_table
class WeightHdfData : public morph::HdfData
{
public:
    WeightHdfData (const std::string fname,
                   const morph::FileAccess _file_access = morph::FileAccess::TruncateWrite,
                   const bool show_hdf_internal_errors = false)
        : HdfData (fname, _file_access, show_hdf_internal_errors) {}

    void read_weighttable (const char* path, std::vector<morph::nn::conn<float>>& vals)
    {
        // Read vectors, then resize/population the vector<conn<float>>.
        std::string pth(path);
        std::string i_path = pth + "_i";
        std::string j_path = pth + "_j";
        std::string w_path = pth + "_w";
        std::vector<unsigned int> i_array (vals.size());
        std::vector<unsigned int> j_array (vals.size());
        std::vector<float> w_array (vals.size());
        this->read_contained_vals (i_path.c_str(), i_array);
        this->read_contained_vals (j_path.c_str(), j_array);
        this->read_contained_vals (w_path.c_str(), w_array);
        if (i_array.size() != j_array.size() || i_array.size() != w_array.size()) {
            vals.resize(0);
            return;
        }
        vals.resize (i_array.size());
        for (unsigned int ii = 0; ii < i_array.size(); ++ii) {
            vals[ii].i = i_array[ii];
            vals[ii].j = j_array[ii];
            vals[ii].w = w_array[ii];
        }
    }
    void add_weighttable (const char* path, const std::vector<morph::nn::conn<float>>& vals)
    {
        // Each conn contains two unsigned ints and one float value. Make 3 vectors and save those.
        std::vector<unsigned int> i_array (vals.size());
        std::vector<unsigned int> j_array (vals.size());
        std::vector<float> w_array (vals.size());
        for (unsigned int ii = 0; ii < vals.size(); ++ii) {
            i_array[ii] = vals[ii].i;
            j_array[ii] = vals[ii].j;
            w_array[ii] = vals[ii].w;
        }
        std::string pth(path);
        std::string i_path = pth + "_i";
        std::string j_path = pth + "_j";
        std::string w_path = pth + "_w";
        this->add_contained_vals (i_path.c_str(), i_array);
        this->add_contained_vals (j_path.c_str(), j_array);
        this->add_contained_vals (w_path.c_str(), w_array);
    }
};
