#include <iostream>
#include <linalg.h>
#include <decomp.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <fstream>
#include "opencv2/opencv.hpp"

const int sparsity = 16;
const float eps = 0.001;
const float lambda = 0.00001;
const int sline = 8;
const int scol = 8;
const int patchesm = sline*scol*3;

const int dictm = 96;
const int dictn = 192;

int main(int argc, char *  argv[]){
    std::cout << "Hello" << std::endl;
    auto cap = cv::VideoCapture(0);

    if (!cap.isOpened()){
        std::cout << "Nao conseguiu abrir a webcam" << std::endl;
        return -1;
    }

    // --------------------------------------------------------------------------------
    // Carrega o dicionário
    ifstream fdict;
    fdict.open("dl8_ycbcr_ds96_720pmoria.txt");
    if (!fdict.is_open()){
        std::cerr << "Erro carregando dicionario" << std::endl;
        return -1;
    }
    Matrix<float> D(dictm, dictn);
    for (int i=0; i<dictm; i++){
        for (int j=0; j<dictn; j++){
            fdict >> D(i,j);
        }
    }
    fdict.close();
    Matrix<float> Dt;
    D.transpose(Dt);
    // --------------------------------------------------------------------------------

    auto image = cv::imread("/home/eduardo/Imagens/720pMoria.png",cv::IMREAD_UNCHANGED);
    if (image.channels() == 4){
        cv::Mat chans[4];
        cv::split(image, chans);
        for (int i=0; i<3; i++){
            chans[i] = 255 - chans[3] + (chans[3].mul(chans[i])) / 255;
        }
        cv::merge(chans, 3, image);
    }
    // cv::Mat image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // std::cout << image.at<cv::Vec4b>(0,0) << std::endl;
    // std::cout << image.at<cv::Vec4b>(0,1) << std::endl;
    
    int patchesn = (image.rows / sline) * (image.cols / scol);
    Matrix<float> im(patchesm, patchesn);
    std::cout << im.m() << " x " << im.n() << std::endl;

    OMPSolver<float> solver(patchesn, Dt, sparsity, eps, lambda, 4);

    double mx, mn;
    cv::minMaxIdx(image, &mn, &mx);
    std::cout << mn << " " << mx << std::endl;

    auto tic = std::chrono::system_clock::now();
    cv::Mat img;
    // std::cout << image.at<cv::Vec3b>(0,0) << std::endl;
    //std::cout << img.at<cv::Vec3f>(0,0) << std::endl;
    cv::cvtColor(image, img, cv::COLOR_RGB2YCrCb);
    img.convertTo(img, CV_32FC3, 1/255.);
    //std::cout << img.at<cv::Vec3f>(0,0) << std::endl;

    int col = 0;
    for (int i=0; i<img.rows; i+=sline){
        for (int j=0; j<img.cols; j+=scol){
            int c=0, a=0, b=0;
            Vector<float> vec;
            im.refCol(col, vec);
            while(c<sline*scol*3){
                auto refmat = img.at<cv::Vec3f>(i + a, j + b);
                vec[c] = refmat[0]; c++;
                vec[c] = refmat[2]; c++;
                vec[c] = refmat[1]; c++;
                a = ++b == scol ? b=0, a+1 : a;
            }
            col++;
        }
    }

    SpMatrix<float> ret;
    solver.solve(im);
    // omp(im, Dt, ret, &sparsity, &eps, &lambda);
    // solver.solve(im);
    // solver.solve(im);
    // solver.solve(im, ret);
    // solve_omp(im, solver, ret);

    auto tac = std::chrono::system_clock::now();
    auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    printf("Elapsed: %ldms\n", tictac);

    // omp(im, Dt, ret, &sparsity, &eps, &lambda);

    std::cout << ret.m() << " x " << ret.n() << std::endl;
    std::cout << Dt.m() << " x " << Dt.n() << std::endl;
    
    tic = std::chrono::system_clock::now();
    solver.getResults(ret);
    Matrix<float> result;
    Dt.mult(ret, result);

    // std::cout << result.m() << " x " << result.n() << std::endl;
    // std::cout << img.at<cv::Vec3f>(7,7) << std::endl;
    // std::cout << im(189,0) << ", " << im(191,0) << ", " << im(190,0) << ", " << std::endl;
    // std::cout << result(189,0) << ", " << result(191,0) << ", " << result(190,0) << ", " << std::endl;

    // -- Salvar resultado
    img.setTo(0);
    col = 0;
    for (int i=0; i<img.rows; i+=sline){
        // auto imgrow = img.rowRange(i, i+sline);
        for (int j=0; j<img.cols; j+=scol){
            // auto refmat = imgrow.colRange(j, j+scol).ptr<float>(0);
            int c=0;
            Vector<float> vec;
            result.refCol(col, vec);
            while(c<sline*scol*3){
                img.at<cv::Vec3f>(i + c/24, j + (c/3)%8) = cv::Vec3f({vec[c], vec[c+2], vec[c+1]});
                c+=3;
            }
            col++;
        }
    }

    img.convertTo(image, CV_8UC3, 255);
    cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);

    tac = std::chrono::system_clock::now();
    tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    printf("Elapsed: %ldms\n", tictac);

    cv::imwrite("generated.png", image);
    
    std::cout << "Conseguiu abrir a webcam" << std::endl;
    cv::Mat frame;
    // cv::namedWindow("WebCam");
    for(;;){
        cap >> frame;
        // std::cout << frame.channels() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(42));
    }
    return 0;
}
