#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char** argv) {

  Mat image;
  image = imread("sample.jpeg" , CV_LOAD_IMAGE_COLOR);

  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }

  namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  imshow( "Display window", image );

  waitKey(0);
  return 0;
}
