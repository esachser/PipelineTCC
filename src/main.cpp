#define ENABLEPRINT

#include <iostream>
#include <linalg.h>
#include <decomp.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
//#include "ompsolveropencv.h"

// #define ENABLEPRINT

#ifdef ENABLEPRINT
#define printval(header, divisor, value, unidade) {std::cout << header << divisor << value << unidade << std::endl;}
#else
#define printval(header, divisor, value, unidade)
#endif

const int sparsity = 5;
const float eps = 0.001;
const float lambda = 0.00001;
const int sline = 4;
const int scol = 4;
const int patchesm = sline*scol*3;

const int dictm = 16;
const int dictn = sline*scol*3;

double getPSNR(cv::Mat& I1, cv::Mat& I2);

int main(int argc, char *  argv[]){
    // std::cout << "Hello" << std::endl;
    // --------------------------------------------------------------------------------
    // Carrega o dicionário
    ifstream fdict;
    fdict.open("dl4_ycbcr_ds16_720pBunny2.txt");
    // fdict.open("dl8_ycbcr_ds64_720ped.txt");
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

    // auto image = cv::imread("/home/eduardo/Imagens/720pMoria.png",cv::IMREAD_UNCHANGED);
    // if (image.channels() == 4){
    //     cv::Mat chans[4];
    //     cv::split(image, chans);
    //     for (int i=0; i<3; i++){
    //         chans[i] = 255 - chans[3] + (chans[3].mul(chans[i])) / 255;
    //     }
    //     cv::merge(chans, 3, image);
    // }
    auto cap = cv::VideoCapture("../Videos/BigBuckBunny.avi");
    // auto cap = cv::VideoCapture("../Videos/ed_1024.avi");

    if (!cap.isOpened()){
        std::cout << "Nao conseguiu abrir a webcam" << std::endl;
        return -1;
    }
    cv::Mat image;
    // cv::imwrite("generated.png", image);
    
    std::cout << "Conseguiu abrir o video escolhido" << std::endl;
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
    // cap.set(cv::CAP_PROP_POS_FRAMES, 2400);

    cap >> image;
    int patchesn = (image.rows / sline) * (image.cols / scol);
    Matrix<float> im(patchesm, patchesn);
    std::cout << im.m() << " x " << im.n() << std::endl;
    std::cout << image.size << std::endl;

    OMPSolver<float> solver(patchesn, Dt, sparsity, eps, lambda, -1);
    cv::Mat image_result;
    cv::namedWindow("WebCam");
    cv::namedWindow("Gerada");
    // cap.set(cv::CAP_PROP_POS_FRAMES, 15000);
    cv::Mat res;
    for(;;){
        // printval("Image size", ": ", image.size, "");
        cv::imshow("WebCam", image);

        // getchar();

        auto tic = std::chrono::system_clock::now();
        cv::Mat img;
        cv::cvtColor(image, res, cv::COLOR_BGR2YCrCb);
        res.convertTo(img, CV_32FC3, 1/255.);

        int col = 0;
        for (int i=0; i<img.rows-sline+1; i+=sline){
            for (int j=0; j<img.cols-scol+1; j+=scol){
                int c=0, a=0, b=0;
                Vector<float> vec;
                im.refCol(col, vec);
                while(c<patchesm){
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
        // lasso(im, Dt, ret, sparsity, lambda);
        // solver.transform0(65535);
        // solver.roundValues();

        auto tac = std::chrono::system_clock::now();
        auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
        // printf("Elapsed encode: %ldms\n", tictac);
        printval("Encode Time", ": ", tictac, "ms");

        // Matrix<float> mtest(patchesm, 1);
        // Vector<float> vdata;
        // SpMatrix<float> tresult;
        // mtest.refCol(0, vdata);
        // im.copyCol(0, vdata);
        // tic = std::chrono::system_clock::now();
        // omp(mtest, Dt, tresult, &sparsity, &eps, &lambda);
        // lasso(mtest, Dt, tresult, sparsity, lambda);
        // ist(mtest, Dt, tresult, eps, constraint_type::L2ERROR);
        
        // tac = std::chrono::system_clock::now();
        // tictac = std::chrono::duration_cast<std::chrono::microseconds>(tac-tic).count();
        // printf("Elapsed encode: %ldms\n", tictac);
        // printval("OMP Time", ": ", tictac, "us");
        
        tic = std::chrono::system_clock::now();
        solver.getResults(ret);
        Matrix<float> result;
        Dt.mult(ret, result);

        // -- Salvar resultado
        img.setTo(0);
        col = 0;
        for (int i=0; i<img.rows-sline+1; i+=sline){
            // auto imgrow = img.rowRange(i, i+sline);
            for (int j=0; j<img.cols-scol+1; j+=scol){
                // auto refmat = imgrow.colRange(j, j+scol).ptr<float>(0);
                int c=0;
                Vector<float> vec;
                result.refCol(col, vec);
                while(c<patchesm){
                    img.at<cv::Vec3f>(i + c/(scol*3), j + (c/3)%sline) = cv::Vec3f({vec[c], vec[c+2], vec[c+1]});
                    c+=3;
                }
                col++;
            }
        }

        img.convertTo(image_result, CV_8UC3, 255);
        cv::cvtColor(image_result, image_result, cv::COLOR_YCrCb2BGR);

        tac = std::chrono::system_clock::now();
        tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
        // printf("Elapsed retrieve: %ldms\n", tictac);
        printval("Decode Time", ": ", tictac, "ms\n");

        cv::imshow("Gerada", image_result);
        auto psnr = getPSNR(image, image_result);
        // printval("PSNR", ": ", psnr, "dB");
        std::cout << psnr << std::endl;
        // if(psnr < 38) cv::waitKey(0);
        

        if (cv::waitKey(1) >= 0) break;
        if (!cap.read(image)) break;
    }
    // if (cap.isOpened())
    //     cap.release();
    // cv::destroyAllWindows();
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