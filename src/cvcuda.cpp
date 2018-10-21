#include <iostream>

// #include "opencv2/opencv_modules.hpp"
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>

// #if defined(HAVE_OPENCV_CUDACODEC)

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>

#include <eigen3/Eigen/Eigen>

#include "dynlink_nvcuvid.cpp"



const int sparsity = 5;
const float eps = 0.0001;
const float lambda = 0.00001;
const int sline = 4;
const int scol = 4;
const int patchesm = sline*scol*3;

const int dictm = 16;
const int dictn = sline*scol*3;


int main(int argc, const char* argv[])
{
    if (argc != 2)
        return -1;

    // -------------------------------------------------------------------------------
    // Carregamento do dicionário
    std::ifstream fdict;
    // fdict.open("dl4_rgb_ds16_720pBunny.txt");
    fdict.open("dl4_rgb_ds16_720ped.txt");    
    if (!fdict.is_open()){
        std::cerr << "Erro carregando dicionario" << std::endl;
        return -1;
    }
    Eigen::MatrixXf D(dictm, dictn);
    for (int i=0; i<dictm; i++){
        for (int j=0; j<dictn; j++){
            fdict >> D(i,j);
        }
    }
    fdict.close();
    Eigen::MatrixXf Dt(dictn, dictm);
    // D.transpose(Dt);
    Dt << D.transpose();
    // -------------------------------------------------------------------------------


    const std::string fname(argv[1]);

    cv::namedWindow("GPU", cv::WINDOW_OPENGL);
    cv::cuda::setGlDevice();

    cuvidInit(1);
    cv::cuda::GpuMat d_frame;
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);

    cv::TickMeter tm;
    std::vector<double> gpu_times;

    int gpu_frame_count=0;

    for (;;)
    {
        tm.reset(); tm.start();
        if (!d_reader->nextFrame(d_frame))
            break;
        tm.stop();
        gpu_times.push_back(tm.getTimeMilli());
        gpu_frame_count++;

        cv::imshow("GPU", d_frame);

        if (cv::waitKey(1) > 0)
            break;
    }

    if (!gpu_times.empty())
    {
        std::cout << std::endl << "Results:" << std::endl;
        std::sort(gpu_times.begin(), gpu_times.end());
        double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
        std::cout << "GPU : Avg : " << gpu_avg << " ms FPS : " << 1000.0 / gpu_avg << " Frames " << gpu_frame_count << std::endl;
    }

    return 0;
}

// #else

// int main()
// {
//     std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
//     return 0;
// }

// #endif