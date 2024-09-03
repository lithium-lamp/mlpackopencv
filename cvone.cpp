#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

cv::Mat contour_frame(cv::Mat frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::Mat canny;  
    cv::Canny(blur, canny, 50, 150);
    return canny;
}


int main() {
    cv::Mat video_from_facecam;

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Could not open video stream!" << std::endl;
        return -1;
    }

    while (char (cv::waitKey(1)) != 'q') {
        cap >> video_from_facecam;

        if (video_from_facecam.empty()) {
            break; 
        }
        
        imshow("Video Player", video_from_facecam);

        imshow("Contour Frame", contour_frame(video_from_facecam));
    }

    cap.release();
    return 0;
}