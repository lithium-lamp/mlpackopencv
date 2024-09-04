#include <iostream>
#include <mlpack.hpp>
#include <opencv2/opencv.hpp>

int hHighL = 250; //half the highlight distance, arbitrary dimension

int iLowH = 8;
int iHighH = 10;

int iLowS = 79; 
int iHighS = 138;

int iLowV = 127;
int iHighV = 255;

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

cv::Mat contour_frame(cv::Mat frame) {
    cv::Mat imgHSV;

    cvtColor(frame, imgHSV, cv::COLOR_BGR2HSV);
 
    cv::Mat imgThresholded;

    cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded);
      
    cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::dilate(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))); 

    cv::dilate( imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
    cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

    return imgThresholded;
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

    while (char (cv::waitKey(1)) != 'q') {
        int64 t1 = cv::getTickCount();
        double secs = (t1-t0)/cv::getTickFrequency();

        cap >> video_from_facecam;

        if (video_from_facecam.empty()) {
            break; 
        }

        if (int(secs) == 10) { //triggers once every ~10 seconds
            t0 = t1;
            
            //arma::mat converted = cvToArma(contour_frame(video_from_facecam));
        
            //mlModel(converted);
        }

        cv::Mat imgThresholded = contour_frame(video_from_facecam);

        cv::Moments oMoments = cv::moments(imgThresholded);

        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

        cv::Mat thresh_color;

        cv::cvtColor(imgThresholded, thresh_color, cv::COLOR_GRAY2RGB);

        cv::Mat out;

        cv::bitwise_and(video_from_facecam, thresh_color, out);

        cv::Mat subImg = cv::Mat::zeros(hHighL * 2, hHighL * 2, CV_64F);

        if (dArea > 10000) {
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;

            posY -= 100; //arbitrary offset
                
            if (posX >= 0 && posY >= 0) {
                const int x1 = posX - hHighL;
                const int x2 = posX + hHighL;

                const int y1 = posY - hHighL;
                const int y2 = posY + hHighL;

                if (x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0) {
                    
                }
                if (x1 > imgTmp.size().width || x2 > imgTmp.size().width || y1 > imgTmp.size().height || y2 > imgTmp.size().height) {
                    
                }
                else {
                    subImg = video_from_facecam(cv::Range(y1, y2), cv::Range(x1, x2));
                    //cv::Mat subImg = out(cv::Range(y1, y2), cv::Range(x1, x2));
                }
            }
        }



        imshow("Highlighted", subImg);

        //imshow("Contour Frame", imgThresholded);

        //imshow("Highlighted", video_from_facecam + imgLines);
    }

    cap.release();

    return 0; 
}