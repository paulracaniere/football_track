#ifndef FOOTBALL_TRACK_MAIN_HPP
#define FOOTBALL_TRACK_MAIN_HPP

#include <opencv2/opencv.hpp>
#include "image.h"
#include <string>
#include <queue>

using namespace cv;
using namespace std;

// Cuts the image horizontally
void cut_horizontal(Image<uchar>& I, int up, int down);

// Computes "distance to void" with succesive erosions
vector<int> dist_to_void(const Image<int>& cc_im, int ccN);

// Removes connected components which have a max dist_to_void below threshold th
void remove_cc(const Image<uchar>& src, Image<uchar>& dst, const Image<int>& cc_im, const vector<int>& dists, int th);

// Shows the image in a miniature
template<typename T>
void imshow2(string str, const Image<T>& src){
    Size display_size(1000, 600);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}

void remove_cc_dim(const Image<uchar>& src, Image<Vec3b>& dst, const Image<int>& cc_im, const vector<float>& dims, float th);

#endif //FOOTBALL_TRACK_MAIN_HPP
