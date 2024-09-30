#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void gradientLength(Mat& src, Mat& dst) {
    Mat img_Gx, img_Gy;
    Sobel(src, img_Gx, CV_32FC1, 1, 0);
    Sobel(src, img_Gy, CV_32FC1, 0, 1);
    magnitude(img_Gx, img_Gy, dst);
}

void gradientAngle(Mat& src, Mat& dst) {
    Mat img_Gx, img_Gy;
    Sobel(src, img_Gx, CV_32FC1, 1, 0);
    Sobel(src, img_Gy, CV_32FC1, 0, 1);
    phase(img_Gx, img_Gy, dst);
}


void nonMaximumSuppression(Mat& matr_gradient, Mat& img_angles) {
    Mat img_border = matr_gradient.clone();
    for (int i = 1; i < matr_gradient.rows - 1; i++) {
        for (int j = 1; j < matr_gradient.cols - 1; j++) {
            float angle = img_angles.at<float>(i, j);
            float gradient = matr_gradient.at<float>(i, j);
            if (i == 0 || i == matr_gradient.rows - 1 || j == 0 || j == matr_gradient.cols - 1) {
                img_border.at<float>(i, j) = 0;
            }
            else {
                int x_shift = 0;
                int y_shift = 0;
                if (angle == 0 || angle == 4) {
                    x_shift = 0;
                }
                else if (angle > 0 && angle < 4) {
                    x_shift = 1;
                }
                else {
                    x_shift = -1;
                }
                if (angle == 2 || angle == 6) {
                    y_shift = 0;
                }
                else if (angle > 2 && angle < 6) {
                    y_shift = -1;
                }
                else {
                    y_shift = 1;
                }
                bool is_max = gradient >= matr_gradient.at<float>(i + y_shift, j + x_shift) && gradient >= matr_gradient.at<float>(i - y_shift, j - x_shift);
                img_border.at<float>(i, j) = is_max ? gradient : 0;
            }
        }
    }
    imshow("img_border", img_border);
}

int main(int argc, char** argv) {

    Mat imgCanny;

    setlocale(LC_ALL, "Russian");
    std::cout << "Введите имя файла изображения: ";
    std::string "1.jpg";
    std::cin >> "1.jpg";

    std::cout << "Ищем " + 1.jpg << std::endl;
    Mat src = imread(1.jpg, IMREAD_GRAYSCALE);

    Mat gradLength(src.size(), CV_32FC1);
    Mat gradAngle(src.size(), CV_32FC1);
    Mat doubleFiltered(src.size(), CV_8UC1);

    gradientLength(src, gradLength);
    gradientAngle(src, gradAngle);
    nonMaximumSuppression(gradLength, gradAngle);
    Canny(src, imgCanny, 50, 150);

    imshow("Original_Image", src);
    imshow("Matr_Gradient", gradLength / 255.0f);
    imshow("Matr_Angle", gradAngle / CV_PI / 2.0f);
    imshow("Double_Filtration", imgCanny);

    waitKey(0);

    return 0;
}
