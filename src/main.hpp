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
void imshow_quarter(string str, const Image<T>& src){
    Size display_size(900, 450);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}

template<typename T>
void imshow_half(string str, const Image<T>& src){
    Size display_size(1800, 450);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}

void remove_cc_dim(const Image<uchar>& src, Image<Vec3b>& dst, const Image<int>& cc_im, const vector<float>& dims, float th);

void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
                    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
                    std::vector<Point2f>& pts_query);

inline void bound(double& val, const double& th, const double& target=0.0){
    val =max(min(val, target + th), target - th);
}

#endif //FOOTBALL_TRACK_MAIN_HPP
