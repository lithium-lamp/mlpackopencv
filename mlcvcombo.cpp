#include <iostream>
#include <mlpack.hpp>
#include <opencv2/opencv.hpp>


int hHighL = 300; //half the highlight distance, arbitrary dimension

int iLowH = 2;
int iHighH = 12;

int iLowS = 77; 
int iHighS = 153;

int iLowV = 93;
int iHighV = 237;

arma::mat cvToArma(const cv::Mat& m) {
    arma::mat arma_mat( reinterpret_cast<double*>(m.data), m.rows, m.cols );

    return arma_mat;
}

void mlModel(const arma::mat& data) { //ml model to be called after image has been properly formatted
    mlpack::neighbor::NeighborSearch<mlpack::neighbor::NearestNeighborSort, 
        mlpack::metric::ManhattanDistance> nn(data);
  
    arma::Mat<size_t> neighbors;
    arma::mat distances;
  
    nn.Search(1, neighbors, distances);

    for (size_t i = 0; i < neighbors.n_elem; ++i) { 
        std::cout << "Nearest neighbor of point " << i << " is point "
            << neighbors[i] << " and the distance is " 
            << distances[i] << ".\n"; 
    }
}

cv::Mat contour_frame(const cv::Mat& frame) {
    cv::Mat imgHSV;

    cvtColor(frame, imgHSV, cv::COLOR_BGR2HSV);
 
    cv::Mat imgThresholded;

    cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded);
      
    cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))); 

    cv::dilate(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
    cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

    return imgThresholded;
}

bool createPartialImage(const cv::Mat& video_from_facecam, const cv::Mat& imgThresholded, cv::Mat& subimg1, cv::Mat& subimg2, cv::Mat& subimg3) {    
    cv::Moments oMoments = cv::moments(imgThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    if (dArea <= 10000)
        return false;

    int posX = dM10 / dArea;
    int posY = dM01 / dArea;

    posY -= 100; //arbitrary offset, adjusting for neck area relative to head.

    const int x1 = posX - hHighL;
    const int x2 = posX + hHighL;

    const int y1 = posY - hHighL;
    const int y2 = posY + hHighL;

    if (x1 <= 0 || y1 <= 0)
        return false;
    
    if (x2 >= video_from_facecam.size().width || y2 >= video_from_facecam.size().height)
        return false;

    cv::Mat thresh_color;

    cv::cvtColor(imgThresholded, thresh_color, cv::COLOR_GRAY2RGB);

    //for obtaining area as seen by mask
    cv::Mat partialmask;

    cv::bitwise_and(video_from_facecam, thresh_color, partialmask);

    subimg1 = video_from_facecam(cv::Range(y1, y2), cv::Range(x1, x2));
    subimg2 = partialmask(cv::Range(y1, y2), cv::Range(x1, x2));
    subimg3 = thresh_color(cv::Range(y1, y2), cv::Range(x1, x2));

    return true;
}

int main() { 
    cv::Mat video_from_facecam;

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Could not open video stream!" << std::endl;
        return -1;
    }

    cv::namedWindow("Control", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("LowH", "Control", &iLowH, 179);
    cv::createTrackbar("HighH", "Control", &iHighH, 179);

    cv::createTrackbar("LowS", "Control", &iLowS, 255);
    cv::createTrackbar("HighS", "Control", &iHighS, 255);

    cv::createTrackbar("LowV", "Control", &iLowV, 255);
    cv::createTrackbar("HighV", "Control", &iHighV, 255);

    int64 t0 = cv::getTickCount();

    cv::Mat imgTmp;
    cap.read(imgTmp); 

    cv::Mat subimg1;
    cv::Mat subimg2;
    cv::Mat subimg3;

    while (char (cv::waitKey(1)) != 'q') {
        int64 t1 = cv::getTickCount();
        double secs = (t1-t0)/cv::getTickFrequency();

        cap >> video_from_facecam;

        if (video_from_facecam.empty()) {
            break; 
        }

        cv::Mat imgThresholded = contour_frame(video_from_facecam);

        bool createdSuccess = false;

        if (createPartialImage(video_from_facecam, imgThresholded, 
                                    subimg1, subimg2, subimg3)) {
            createdSuccess = true;
        }

        if (int(secs) >= 1) { //triggers once every ~10 seconds
            t0 = t1;

            if (createdSuccess) {
                int val = std::rand() % 10;
                imwrite("trainingdata/cropped/" + std::to_string(val) + ".png", subimg1);
                imwrite("trainingdata/partialmask/" + std::to_string(val) + ".png", subimg2);
                imwrite("trainingdata/mask/" + std::to_string(val) + ".png", subimg3);
            }
        }

        imshow("Cropped", subimg1);

        imshow("Contour Frame", imgThresholded);
    }

    cap.release();

    return 0; 
}