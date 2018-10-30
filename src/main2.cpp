// #define ENABLEPRINT
#define EIGEN_USE_MKL_ALL
// #define MKL_DIRECT_CALL

#define ENABLEPRINT

#include <iostream>
#include <time.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include "ompsolvereigen.h"
#include "ompsolvercuda.h"
// #include "ksvd.hpp"

#ifdef ENABLEPRINT
#define printval(header, divisor, value, unidade) {std::cout << header << divisor << value << unidade << std::endl;}
#else
#define printval(header, divisor, value, unidade)
#endif

const int sparsity = 5;
const float eps = 0.0001;
const float lambda = 0.00001;
const int sline = 4;
const int scol = 4;
const int patchesm = sline*scol*3;

const int dictm = 16;
const int dictn = sline*scol*3;

const int PLUS = '+';
const int MINUS = '-';

double getPSNR(cv::Mat& I1, cv::Mat& I2);

int main(int argc, char *  argv[]){
    // std::cout << "Hello" << std::endl;
    // --------------------------------------------------------------------------------
    // Carrega o dicionário
    std::ifstream fdict;
    // fdict.open("dl4_rgb_ds16_Bunny.txt");
    fdict.open("dl4_rgb_ds16_sintel.txt");
    // fdict.open("dl4_rgb_ds16_720ped.txt");    
    if (!fdict.is_open()){
        std::cerr << "Erro carregando dicionario" << std::endl;
        return -1;
    }
    Eigen::MatrixXf D(dictm, dictn);
    Eigen::MatrixXf Dt(dictn, dictm);
    for (int i=0; i<D.rows(); i++){
        for (int j=0; j<D.cols(); j++){
            fdict >> D(i,j);
        }
    }
    fdict.close();
    Dt << D.transpose();

    // --------------------------------------------------------------------------------

    // auto cap = cv::VideoCapture("../Videos/stefan_sif.y4m");
    // auto cap = cv::VideoCapture("../Videos/station2_1080p25.y4m");
    auto cap = cv::VideoCapture("../Videos/sintel-1024-surround.mp4");
    // auto cap = cv::VideoCapture("../Videos/ed_1024.avi");
    // auto cap = cv::VideoCapture("../Videos/park_joy_444_720p50.y4m");

    if (!cap.isOpened()){
        std::cout << "Nao conseguiu abrir a webcam" << std::endl;
        return -1;
    }
    cv::Mat image;
    // cv::imwrite("generated.png", image);
    
    std::cout << "Conseguiu abrir o video escolhido" << std::endl;
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
    // cap.set(cv::CAP_PROP_POS_FRAMES, 300*24);

    cap >> image;

    // Calcula quanto terá de ter de bordas
    int rsize = image.rows / sline;
    if (image.rows%sline > 0) rsize++;
    int csize = image.cols / sline;
    if (image.cols%sline > 0) csize++;

    int rborder = image.rows % sline;
    int cborder = image.cols % scol;

    int patchesn = (rsize) * (csize);
    Eigen::MatrixXf im(patchesm, patchesn);
    Eigen::MatrixXf imant(patchesm, patchesn);
    Eigen::VectorXi pequal(patchesn);
    std::cout << im.rows() << " x " << im.cols() << std::endl;
    std::cout << image.size << std::endl;

    // IHTSolverEigen solver(patchesn, Dt, sparsity, eps, lambda, -1);
    // OMPSolverEigen solver(patchesn, Dt, sparsity, eps, lambda, -1);
    OMPSolverCUDAEigen solver(patchesn, Dt, sparsity, eps, lambda, -1);
    cv::Mat image_result;
    cv::namedWindow("WebCam");
    cv::namedWindow("Gerada");
    // Eigen::MatrixXf ret(patchesn, dictm);
    Eigen::MatrixXf ret(dictm, patchesn);
    Eigen::MatrixXf result(patchesm, patchesn);
    cv::Mat res;

    image.copyTo(image_result);

    cv::Mat3f anterior;
    cv::Mat3f resanterior;
    anterior.copySize(image);
    anterior.setTo(0);
    resanterior.copySize(image);
    resanterior.setTo(cv::Vec3f({0,0,0}));

    // diffimage.copySize(image);
    // diffimage.setTo(cv::Vec3f({0,0,0}));
    cv::Mat diffimage(cv::Size(csize*scol, rsize*sline), CV_32FC3);
    diffimage.setTo(cv::Vec3f({0,0,0}));

    int quality = sparsity;
    im.setZero();
    imant.setOnes();

    for(;;){
        // printval("Image size", ": ", image.size, "");
        cv::imshow("WebCam", image);

        // getchar();

        auto tic = std::chrono::system_clock::now();
        cv::Mat img;
        // cv::cvtColor(image, res, cv::COLOR_BGR2RGB);
        // image.convertTo(img, CV_32FC3, 1/255.);

        // diffimage = img - resanterior;
        // diffimage = img;
        // diffimage = image / 255.f;
        // img.copyTo(anterior);
        // pequal.setOnes();
        // gmat.upload(image);
        cv::copyMakeBorder(image, img, 0, rborder, 0, cborder, cv::BORDER_REPLICATE, 0);      

        int col = 0;
        // for (int i=0; i<img.rows-sline+1; i+=sline){
        //     for (int j=0; j<img.cols-scol+1; j+=scol, col++){
        //         int c=0, a=0, b=0;
        //         // Vector<float> vec;
        //         auto vec = im.col(col);
        //         auto vecant = imant.col(col);
        //         while(c<patchesm){
        //             auto refmat = img.at<cv::Vec3b>(i + a, j + b);
        //             vec[c] = refmat[2] / 255.f; c++;
        //             vec[c] = refmat[1] / 255.f; c++;
        //             vec[c] = refmat[0] / 255.f; c++;
        //             a = ++b == scol ? b=0, a+1 : a;
        //             printval(a, " ", b, "");
        //         }
        //         pequal[col] = vec.isApprox(vecant, 0.004);
        //         exit(0);
        //     }
        // }
        // im /= 255.f;
        // pequal << im.is
        
        // solver.solve(im, pequal);
        solver.solve(img.data, img.rows, img.cols, sline, scol);
        // solver2.solve(im);
        // lasso(im, Dt, ret, sparsity, lambda);
        // solver.transform0(65535);
        // solver.roundValues();
        // im.transposeInPlace();
        // OMPEncodeSignal<Eigen::MatrixXf, Eigen::MatrixXf>(ret, D, im.transpose(), sparsity, default_encoding_parameters);

        auto tac = std::chrono::system_clock::now();
        auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
        printval("Encode Time", ": ", tictac, "ms");

        imant << im;

        tic = std::chrono::system_clock::now();
        // ret.fill(0.0f);
        // solver.getResults(ret, quality);
        // // Matrix<float> result;
        // // Dt.mult(ret, result);
        // // result << Dt * ret.transpose();
        // // auto max = ret.maxCoeff();
        // // auto min = ret.minCoeff();
        // // printval("Max", ":", max, "");
        // // printval("Min", ":", min, "");
        // result << Dt * ret;
        solver.decode(result, quality);

        // -- Salvar resultado
        // img.setTo(0);
        col = 0;
        
        for (int i=0; i<diffimage.rows-sline+1; i+=sline){
            // auto imgrow = diffimage.rowRange(i, i+sline);
            for (int j=0; j<diffimage.cols-scol+1; j+=scol, col++){
                // auto refmat = imgrow.colRange(j, j+scol).ptr<float>(0);
                int c=0;
                // Vector<float> vec;
                // result.refCol(col, vec);
                auto vec = result.col(col);
                // std::cout << "Aqui" << std::endl;
                while(c<patchesm){
                    diffimage.at<cv::Vec3f>(i + c/(scol*3), j + (c/3)%sline) = cv::Vec3f({vec[c+2], vec[c+1], vec[c]});
                    // diffimage.at<cv::Vec3f>(i + (c/3)%(scol), j + (c/(sline*3))) = cv::Vec3f({vec[c+2], vec[c+1], vec[c]});
                    c+=3;
                }
            }
        }


        resanterior = diffimage(cv::Rect(0, 0, image.cols, image.rows));
        // resanterior += diffimage;
        // resanterior.adjustROI(0, image.rows, 0, image.cols);
        resanterior.convertTo(image_result, CV_8UC3, 255);
        // cv::cvtColor(image_result, image_result, cv::COLOR_RGB2BGR);

        tac = std::chrono::system_clock::now();
        tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
        // printf("Elapsed retrieve: %ldms\n", tictac);
        printval("Decode Time", ": ", tictac, "ms\n");

        cv::imshow("Gerada", image_result);
        auto psnr = getPSNR(image, image_result);
        // printval("PSNR", ": ", psnr, "dB");
        printval(psnr, "dB", "", "");
        // if(psnr < 38) cv::waitKey(0);

        auto key = cv::waitKey(1);
        if (key == PLUS) quality = std::min(sparsity, quality+1);
        if (key == MINUS) quality = std::max(1, quality-1);

        if (key>=0) {
            std::cout << "Sparsity: " << quality << std::endl;
            std::cout << "Key: " << key << std::endl;
        }

        if (tolower(key) == 'q') break;
        if (!cap.read(image)) break;
    }

    return 0;
}


double getPSNR(cv::Mat& I1, cv::Mat& I2){
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    
    cv::Scalar s = cv::sum(s1);

    double sse = s.val[0] + s.val[1] + s.val[2];

    if (sse < 1e-10)
        return 0;
    else {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255*255) / mse);
        return psnr;
    }
}