#include <opencv2/opencv.hpp>
#include "image.h"

using namespace cv;
using namespace std;

int main(int argn, char **argc) {
    VideoCapture cap("../data/footdata2.mp4"); // Ouvre la vidéo

    if (!cap.isOpened()) {  // Vérifie l'ouverture
        cerr << "ERREUR : Lecture de vidéo impossible" << endl;
        return -1;
    }

    Image<Vec3b> I;
    cap >> I;

    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(1000); //MOG2 approach
    Image<Vec3b> out;

    while(I.width() > 0 && I.height() > 0) {
//        imshow("I", I);
        pMOG2->apply(I, out);
        cap >> I;
        imshow("out", out);
        waitKey();
    }

    waitKey(0);

    return 0;
}