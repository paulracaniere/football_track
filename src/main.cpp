#include "main.hpp"

int main(int argn, char **argc) {
    VideoCapture cap("../data/footdata2.mp4"); // Ouvre la vidéo
    if (!cap.isOpened()) {  // Vérifie l'ouverture
        cerr << "ERREUR : Lecture de vidéo impossible" << endl;
        return -1;
    }

    Image<Vec3b> I;
    cap >> I;
    Image<Vec3b> field_im(4*I.width(), 2*I.height());  // Image to contain the entire football field

    size_t click_num = 1;
    size_t fpc=5;  // fpc = frame per click

    // Position offset of current image inside field_im
    double x_off_set = field_im.width()-2*I.width();
    double y_off_set = 0.5 * I.height() ;

    // bounds for the homography (needs to be multiplied by fpc)
    float x_off_bound = 15;
    float y_off_bound = 1;
    float scale_bound = 0.0001;
    float skew_bound = 0.1;
    float z_bound = 0.00000001;

    /*
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(100);
    Image<uchar> move_mask, eroded, eroded_dilated;
    Image<Vec3b> dims_epurated(I.width(), I.height());
    Image<int> connected_comps;

    int erosion_type = MORPH_ELLIPSE;
    int erosion_size = 4;
    int dilate_size = erosion_size + 3;
    */

    // Init of warp
    Ptr<AKAZE> akaze = AKAZE::create();
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;

    // Fill field_im with initial image
    Mat initial_warp = (Mat_<float>(3,3) << 1, 0, x_off_set, 0, 1, y_off_set, 0, 0, 1);
    warpPerspective(I, field_im, initial_warp, field_im.size());

    // Mask to exclude the static box on the top left of images
    Image<uchar> akaze_mask(I.width(), I.height());
    for(int x=0; x<akaze_mask.width(); x++){
        for(int y=0; y<akaze_mask.height(); y++){
            akaze_mask(x,y) = (y > 150 || x > 1200 ? uchar(255) : uchar(0));
        }
    }

    // Detect kpts on initial image
    akaze->detectAndCompute(I, akaze_mask, kpts1, desc1);

    while(I.width() > 0 && I.height() > 0) {
        cout << "Click number : " << click_num++ << endl;
        Image<Vec3b> prevI = I;  // used to display the matchings

        for(int i=0; i<fpc; i++) cap >> I;

        // Finding homography
        akaze->detectAndCompute(I, akaze_mask, kpts2, desc2);
        matcher.match(desc1, desc2, matches);
        vector<Point2f> pts1, pts2;
        matches2points(kpts2, kpts1, matches, pts2, pts1);  // Excludes the top-left static box
        Mat H = findHomography(pts2, pts1, RANSAC);

        // bounding homography
        bound(H.at<double>(0,2), x_off_bound*fpc);
        bound(H.at<double>(1,2), y_off_bound*fpc);
        bound(H.at<double>(0,0), scale_bound*fpc, 1.);
        bound(H.at<double>(1,1), scale_bound*fpc, 1.);
        bound(H.at<double>(0,1), skew_bound*fpc);
        bound(H.at<double>(1,0), skew_bound*fpc);
        bound(H.at<double>(2,0), z_bound*fpc);
        bound(H.at<double>(2,1), z_bound*fpc);

        // Taking into account offset
        H.at<double>(0,2) += x_off_set;
        H.at<double>(1, 2) += y_off_set;
        x_off_set = H.at<double>(0,2);
        y_off_set = H.at<double>(1,2);

        warpPerspective(I, field_im, H, field_im.size(), INTER_LINEAR, BORDER_TRANSPARENT);

        imshow_half("field_im", field_im);
        imshow_quarter("I", I);

        /*
        Image<Vec3b> kpt;
        drawKeypoints(I, kpts2, kpt);
        imshow_quarter("kpt", kpt);

        Image<uchar> M(2 * I.cols, I.rows);
        drawMatches(prevI, kpts1, I, kpts2, matches, M, Scalar::all(-1), Scalar::all(-1), vector<char>(), 4);
        imshow_quarter("matches", M);
        */

        kpts1 = kpts2;
        desc1 = desc2;

        /*
        pMOG2->apply(I, move_mask);
        imshow_quarter("i",I);
        imshow_quarter("move_mask", move_mask);
        cut_horizontal(move_mask, 0, 100);
        cut_horizontal(move_mask, 978, 1078); // not 1080 because of erosion ?
        Mat erode_ker = getStructuringElement( erosion_type,
                                                   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                                   Point( erosion_size, erosion_size ) );
        Mat dilate_ker = getStructuringElement( erosion_type,
                                               Size( 2*dilate_size + 1, 2*dilate_size+1 ),
                                               Point( dilate_size, dilate_size ) );
        erode( move_mask, eroded, erode_ker );
        dilate( eroded, eroded_dilated, dilate_ker );
        Mat centroids;
        Image<int> stats;
        int ccN = connectedComponentsWithStats(eroded_dilated, connected_comps, stats, centroids);  // Number of connected components
        vector<float> dimensions;
        for(int i=0; i<ccN; i++){
            dimensions.push_back((float) stats(CC_STAT_HEIGHT, i) / (float) stats(CC_STAT_WIDTH, i));
        }
        remove_cc_dim(eroded_dilated, dims_epurated, connected_comps, dimensions, (float) 4.);
        imshow_quarter("dim", dims_epurated);
        imshow_quarter("eroded_dilated", eroded_dilated);
        cout << "Connected components : " <<  ccN << endl;
        */


        waitKey();
        cout << endl;
    }
    waitKey(0);

    return 0;
}

void cut_horizontal(Image<uchar>& I, int up, int down){
    for(int y=up; y<down; y++){
        for(int x=0; x<I.width(); x++){
            I(x,y) = 0;
        }
    }
}

void remove_cc_dim(const Image<uchar>& src, Image<Vec3b>& dst, const Image<int>& cc_im, const vector<float>& dims, float th){
    int count =0;
    for(float v : dims) if(v >= th) count++;
    for(int x=0; x<src.width(); x++){
        for(int y=0; y<src.height(); y++){
            if(src(x,y) != (uchar) 0) dst(x,y) = (dims[cc_im(x,y)] < th ? Vec3b(255,255,255) : Vec3b(0,0,255));
            else dst(x,y) = Vec3b(0,0,0);
        }
    }
    cout << "Eliminated " << count << endl;
}


vector<int> dist_to_void(const Image<int>& cc_im, int ccN){
    cout << "Begin erosino for " << ccN << "connected components" << endl;
    Image<uchar> dists(cc_im.width(), cc_im.height()), prev(cc_im.width(), cc_im.height()), erod;
    for(int x=0; x<cc_im.width(); x++){
        for(int y=0; y<cc_im.height(); y++){
            dists(x,y) = (cc_im(x,y) ? (uchar) 1 : (uchar) 0);
            prev(x,y) = dists(x,y);
        }
    }
    Mat ker = getStructuringElement( MORPH_RECT, Size( 3, 3), Point( 1, 1));
    int erosions =0;
    while(countNonZero(prev) > 0){
        cout << erosions++ << " ";
        erode( prev, erod, ker );
        prev = erod;

        for(int x=0; x<cc_im.width(); x++){
            for(int y=0; y<cc_im.height(); y++){
                dists(x,y) = uchar( int(dists(x,y)) + int(erod(x,y)));
            }
        }
    }
    cout << endl;

    cout << "Number of erosions : " << erosions << endl;
    vector<int> res(ccN,0);
    for(int x=0; x<cc_im.width(); x++){
        for(int y=0; y<cc_im.height(); y++){
            res[cc_im(x,y)] = max(res[cc_im(x,y)], (int) dists(x,y));
        }
    }

    return res;

}

void remove_cc(const Image<uchar>& src, Image<uchar>& dst, const Image<int>& cc_im, const vector<int>& dists, int th){
    int count =0;
    for(int x=0; x<src.width(); x++){
        for(int y=0; y<src.height(); y++){
            dst(x,y) = (dists[cc_im(x,y)] > th ? src(x,y) : (uchar) 0);
            if(dists[cc_im(x,y)] <= th) count ++;
        }
    }
    cout << "Eliminated " << count << endl;
}

void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
                    const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
                    std::vector<Point2f>& pts_query)
{
    for(size_t i = 0; i < matches.size(); i++)
    {
        const DMatch & dmatch = matches[i];
        Point2f dif = query[dmatch.queryIdx].pt - train[dmatch.trainIdx].pt;
        if(dif.x*dif.x + dif.y*dif.y < 100000){
            pts_query.push_back(query[dmatch.queryIdx].pt);
            pts_train.push_back(train[dmatch.trainIdx].pt);
        }
    }
    cout << pts_train.size() << " points kept out of " << train.size() << endl;
}


