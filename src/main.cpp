// #define ENABLEPRINT

#include <iostream>
#include <linalg.h>
#include <decomp.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <fstream>
#include "opencv2/opencv.hpp"

#define ENABLEPRINT

#ifdef ENABLEPRINT
#define printval(header, divisor, value, unidade) {std::cout << header << divisor << value << unidade << std::endl;}
#else
#define printval(header, divisor, value, unidade)
#endif

const int sparsity = 16;
const float eps = 0.001;
const float lambda = 0.00001;
const int sline = 8;
const int scol = 8;
const int patchesm = sline*scol*3;

const int dictm = 96;
const int dictn = 192;

int main(int argc, char *  argv[]){
    // std::cout << "Hello" << std::endl;

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

    // auto image = cv::imread("/home/eduardo/Imagens/720pMoria.png",cv::IMREAD_UNCHANGED);
    // if (image.channels() == 4){
    //     cv::Mat chans[4];
    //     cv::split(image, chans);
    //     for (int i=0; i<3; i++){
    //         chans[i] = 255 - chans[3] + (chans[3].mul(chans[i])) / 255;
    //     }
    //     cv::merge(chans, 3, image);
    // }
    auto cap = cv::VideoCapture("../Videos/ed_1024.avi");

    if (!cap.isOpened()){
        std::cout << "Nao conseguiu abrir a webcam" << std::endl;
        return -1;
    }
    cv::Mat image;
    // cv::imwrite("generated.png", image);
    
    std::cout << "Conseguiu abrir o video escolhido" << std::endl;
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);

    cap >> image;
    int patchesn = (image.rows / sline) * (image.cols / scol);
    Matrix<float> im(patchesm, patchesn);
    std::cout << im.m() << " x " << im.n() << std::endl;
    std::cout << image.size << std::endl;

    OMPSolver<float> solver(patchesn, Dt, sparsity, eps, lambda, 4);
    cv::Mat image_result;
    cv::namedWindow("WebCam");
    cv::namedWindow("Gerada");
    for(;;){
        // printval("Image size", ": ", image.size, "");
        cv::imshow("WebCam", image);

        // getchar();

        auto tic = std::chrono::system_clock::now();
        cv::Mat img;
        cv::cvtColor(image, img, cv::COLOR_BGR2YCrCb);
        img.convertTo(img, CV_32FC3, 1/255.);

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
        solver.transform0(4095);
        solver.roundValues();

        auto tac = std::chrono::system_clock::now();
        auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
        // printf("Elapsed encode: %ldms\n", tictac);
        printval("Encode Time", ": ", tictac, "ms");
        
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
                    img.at<cv::Vec3f>(i + c/24, j + (c/3)%8) = cv::Vec3f({vec[c], vec[c+2], vec[c+1]});
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

        if (cv::waitKey(1) >= 0) break;
        if (!cap.read(image)) break;
    }
    // if (cap.isOpened())
    //     cap.release();
    // cv::destroyAllWindows();
    return 0;
}
