#include <iostream>
#include <mlpack.hpp>
#include <opencv2/opencv.hpp>
  
arma::mat cvToArma(const cv::Mat& m) {
    arma::mat arma_mat( reinterpret_cast<double*>(m.data), m.rows, m.cols );

    return arma_mat;
}

void mlModel(const arma::mat& data) { 
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

    int64 t0 = cv::getTickCount();

    while (char (cv::waitKey(1)) != 'q') {
        int64 t1 = cv::getTickCount();
        double secs = (t1-t0)/cv::getTickFrequency();

        cap >> video_from_facecam;

        if (video_from_facecam.empty()) {
            break; 
        }

        if (int(secs) == 10) { //triggers once every ~10 seconds
            t0 = t1;
            
            arma::mat converted = cvToArma(contour_frame(video_from_facecam));
        
            mlModel(converted);
        }

        imshow("Contour Frame", contour_frame(video_from_facecam));
    }

    cap.release();

    return 0; 
}