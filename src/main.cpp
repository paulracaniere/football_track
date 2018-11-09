#include "main.hpp"

int main(int argn, char **argc) {
    VideoCapture cap("../data/footdata2.mp4"); // Ouvre la vidéo
    if (!cap.isOpened()) {  // Vérifie l'ouverture
        cerr << "ERREUR : Lecture de vidéo impossible" << endl;
        return -1;
    }
    Image<Vec3b> I;
    cap >> I;

    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(100);
    Image<uchar> move_mask, eroded, eroded_dilated;
    Image<Vec3b> dims_epurated(I.width(), I.height());
    Image<int> connected_comps;

    int erosion_type = MORPH_ELLIPSE;
    int erosion_size = 4;
    int dilate_size = erosion_size + 3;
    int frame_num = 0;

    while(I.width() > 0 && I.height() > 0) {
        cout << "Frame number : " << frame_num++ << endl;
        pMOG2->apply(I, move_mask);
        cap >> I;
        imshow2("move_mask", move_mask);
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
        imshow2("dim", dims_epurated);
        imshow2("eroded_dilated", eroded_dilated);
        cout << "Connected components : " <<  ccN << endl;
        cout << endl;
        waitKey();
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


