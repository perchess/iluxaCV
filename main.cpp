#include <QCoreApplication>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <iostream>
#include <Fourier1D.h>

using namespace cv;



//bool isPowerOfTwo (int v)
//{
//    return v && !(v & (v - 1));
//}


//int getOptimalDFTSize(int vecsize)
//{
//    int pow = 0;
//    if (!isPowerOfTwo(vecsize))
//        pow = int(std::log2(vecsize)) + 1;
//    else
//        return vecsize;
//    return 1 << pow;
//}


//void swapSpektr(cv::Mat &magI){
//    // Центр картинки
//    int cx = magI.cols / 2;
//    int cy = magI.rows / 2;
//    // Создаем ROI
//    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
//    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
//    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
//    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

//    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);

//    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//    q2.copyTo(q1);
//    tmp.copyTo(q2);
//}


cv::Mat butterworth(size_t p, size_t q, size_t d_0, size_t n = 1)
{
    cv::Mat img(cv::Size(p, q), CV_32FC1,cv::Scalar(0));
    double d = 0;
    double h = 0;
    for (size_t i = 0; i < img.cols; i++)
        for (size_t j = 0; j < img.rows; j++)
        {
            d = sqrt(pow((i - p/2.0),2) + pow((j - q/2.0), 2));
            img.at<float>(i, j) = 1 / (1 + pow((d/d_0), 2 * n));
        }
    cv::Mat toMerge[] = {img, img};
    cv::merge(toMerge, 2, img);
    return img;
}


cv::Mat perfectFilter(size_t p, size_t q, size_t d_0)
{
    cv::Mat img(cv::Size(p, q), CV_32FC1,cv::Scalar(0));
    double d = 0;
    double h = 0;
    for (size_t i = 0; i < img.cols; i++)
        for (size_t j = 0; j < img.rows; j++)
        {
            d = sqrt(pow((i - p/2.0),2) + pow((j - q/2.0), 2));
            img.at<float>(i, j) = d <= d_0 ? 1 : 0;
        }
    cv::Mat toMerge[] = {img, img};
    cv::merge(toMerge, 2, img);
    return img;
}


void resizeWithZeros(cv::Mat& mat, cv::Size size)
{
    cv::Mat newImg(size, mat.type(),cv::Scalar(0));
    cv::Mat tmpROI(newImg, cv::Rect(0, 0, mat.cols, mat.rows));
    mat.copyTo(tmpROI);
    mat = newImg.clone();
}

cv::Mat createSpectrumHumanRdbl(const cv::Mat& src)
{
    Mat parts[2];

    split(src, parts);
    magnitude(parts[0], parts[1], parts[0]);
    parts[0] += cv::Scalar::all(1);
    log(parts[0], parts[0]);
    cv::normalize(parts[0], parts[0], 0, 1, cv::NormTypes::NORM_MINMAX);
    swapSpektr(parts[0]);
    return parts[0];
}






#define PART 3


int main(int argc, char *argv[])
{
    cv::Mat inputImg = cv::imread("/home/den/Pictures/Bikesgray.jpg", cv::IMREAD_GRAYSCALE);
    inputImg.convertTo(inputImg, CV_32FC1); // Конвертировать во Float

#if PART==2
    cv::Mat inputFourierBI;
    cv::Mat inputFourierHM;
    cv::Mat butterSrc = butterworth(64, 64, 5, 2);
    cv::Mat perfectSrc = perfectFilter(256, 256, 5);
    cv::dft(inputImg, inputFourierBI, cv::DFT_COMPLEX_OUTPUT);
    Fourier1D my_fourier4ik(inputImg);
    inputFourierHM = my_fourier4ik.getFourierImage();

    cv::Mat mulSpectr;
    swapSpektr(inputFourierHM);
    mulSpectr = mulComplex(butterSrc, inputFourierHM);
    swapSpektr(mulSpectr);
    Mat spectr = createSpectrumHumanRdbl(mulSpectr);


    cv::Mat final;
    cv::Mat finalHM(cv::Size(mulSpectr.cols, mulSpectr.rows), CV_32FC1, cv::Scalar(0));;
    cv::dft(mulSpectr, final, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    finalHM = inverseTransformFromSpectrum(mulSpectr);
    cv::normalize(final, final, 0, 1, cv::NormTypes::NORM_MINMAX);
    cv::normalize(finalHM, finalHM, 0, 1, cv::NormTypes::NORM_MINMAX);
    cv::normalize(inputImg, inputImg, 0, 1, cv::NormTypes::NORM_MINMAX);
#endif



#if PART==3
    float sobelXD[]  = {1, 0, -1,
                        2, 0, -2,
                        1, 0, -1};
    Mat sobelX(3, 3, CV_32FC1, &sobelXD);

    float sobelYD[]  = {1,  2,  1,
                        0,  0,  0,
                       -1, -2, -1};
    Mat sobelY(3, 3, CV_32FC1, &sobelYD);

    float laplasD[]  = {0,  1,   0,
                        1,  -4,  1,
                        0,  1,   0};
    Mat laplas(3, 3, CV_32FC1, &laplasD);

    float boxD[]  =    {1, 1, 1,
                        1, 1, 1,
                        1, 1, 1,};
    Mat box(3, 3, CV_32FC1, &boxD);

    cv::Mat inputFourierBI;
    cv::Mat sobelXF;
    cv::Mat sobelYF;
    cv::Mat laplasF;
    cv::Mat boxF;
    cv::Mat inputClone = inputImg.clone();
    cv::Mat final;
    size_t rowSize = inputImg.rows + sobelX.rows - 1;
    size_t colsSize = inputImg.cols + sobelX.cols - 1;

    resizeWithZeros(inputClone, Size(colsSize, rowSize));
    resizeWithZeros(sobelX, Size(colsSize, rowSize));
    resizeWithZeros(sobelY, Size(colsSize, rowSize));
    resizeWithZeros(laplas, Size(colsSize, rowSize));
    resizeWithZeros(box, Size(colsSize, rowSize));
    cv::dft(inputClone, inputFourierBI, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(sobelX, sobelXF, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(sobelY, sobelYF, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(laplas, laplasF, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(box, boxF, cv::DFT_COMPLEX_OUTPUT);
    Mat sobelSpectrum = createSpectrumHumanRdbl(sobelXF);
    Mat laplasSpectrum = createSpectrumHumanRdbl(laplasF);
    Mat boxSpectrum = createSpectrumHumanRdbl(boxF);


//    multiply(inputFourierBI, sobelXF, sobelXResult);
    cv::Mat gx;
    cv::Mat gy;
    cv::Mat l;
    cv::Mat b;
    gx = mulComplex(sobelXF, inputFourierBI);
    gy = mulComplex(sobelYF, inputFourierBI);
    l = mulComplex(laplasF, inputFourierBI);
    b = mulComplex(boxF, inputFourierBI);


    Mat imgSpectrum = createSpectrumHumanRdbl(inputFourierBI);

    cv::dft(b, final, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    resize(final, final, Size(inputImg.cols, inputImg.rows));
    cv::normalize(final, final, 0, 1, cv::NormTypes::NORM_MINMAX);
    cv::normalize(inputImg, inputImg, 0, 1, cv::NormTypes::NORM_MINMAX);

#endif
    while (cv::waitKey(30) != 'q')
    {
#if PART==2
        cv::imshow("output", final);
        cv::imshow("output2", finalHM);
#endif
#if PART==3
        cv::imshow("output", final);
        cv::imshow("sobelSpectrum", sobelSpectrum);
        cv::imshow("imgSpectrum", imgSpectrum);
        cv::imshow("laplasSpectrum", laplasSpectrum);
        cv::imshow("boxSpectrum", boxSpectrum);
#endif
        cv::imshow("inputImg", inputImg);
    }
    return 0;
}
