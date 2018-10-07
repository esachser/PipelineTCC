// #define ENABLEPRINT

#include <iostream>
#include <linalg.h>
#include <decomp.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
//#include "ompsolveropencv.h"

#define ENABLEPRINT

#ifdef ENABLEPRINT
#define printval(header, divisor, value, unidade) {std::cout << header << divisor << value << unidade << std::endl;}
#else
#define printval(header, divisor, value, unidade)
#endif

const int sparsity = 8;
const float eps = 0.001;
const float lambda = 0.00001;
const int sline = 8;
const int scol = 8;
const int patchesm = sline*scol;

const int dictm = 48;
const int dictn = sline*scol;

double getPSNR(cv::Mat& I1, cv::Mat& I2);

int main(int argc, char *  argv[]){
    // std::cout << "Hello" << std::endl;
    // --------------------------------------------------------------------------------
    // Carrega o dicionário
    ifstream fdict;
    fdict.open("dldiff8_y_ds48_720pBunny.txt");
    // fdict.open("dl8_ycbcr_ds64_720ped.txt");
    if (!fdict.is_open()){
        std::cerr << "Erro carregando dicionario" << std::endl;
        return -1;
    }
    Matrix<float> Dy(dictm, dictn);
    for (int i=0; i<dictm; i++){
        for (int j=0; j<dictn; j++){
            fdict >> Dy(i,j);
        }
    }
    fdict.close();
    Matrix<float> Dty;
    Dy.transpose(Dty);

    // ifstream fdict;
    fdict.open("dldiff8_cb_ds48_720pBunny.txt");
    // fdict.open("dl8_ycbcr_ds64_720ped.txt");
    if (!fdict.is_open()){
        std::cerr << "Erro carregando dicionario" << std::endl;
        return -1;
    }
    Matrix<float> Dcb(dictm, dictn);
    for (int i=0; i<dictm; i++){
        for (int j=0; j<dictn; j++){
            fdict >> Dcb(i,j);
        }
    }
    fdict.close();
    Matrix<float> Dtcb;
    Dcb.transpose(Dtcb);

    // ifstream fdict;
    fdict.open("dldiff8_cr_ds48_720pBunny.txt");
    // fdict.open("dl8_ycbcr_ds64_720ped.txt");
    if (!fdict.is_open()){
        std::cerr << "Erro carregando dicionario" << std::endl;
        return -1;
    }
    Matrix<float> Dcr(dictm, dictn);
    for (int i=0; i<dictm; i++){
        for (int j=0; j<dictn; j++){
            fdict >> Dcr(i,j);
        }
    }
    fdict.close();
    Matrix<float> Dtcr;
    Dcr.transpose(Dtcr);
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
    cap.set(cv::CAP_PROP_POS_FRAMES, 480*24);

    cap >> image;
    // image.copyTo(anterior);
    int patchesn = (image.rows / sline) * (image.cols / scol);
    Matrix<float> imy(patchesm, patchesn);
    Matrix<float> imcb(patchesm, patchesn);
    Matrix<float> imcr(patchesm, patchesn);
    // std::cout << im.m() << " x " << im.n() << std::endl;
    std::cout << image.size << std::endl;

    OMPSolver<float> solvery(patchesn, Dty, sparsity, eps, lambda, -1);
    OMPSolver<float> solvercb(patchesn, Dtcb, 4, eps, lambda, -1);
    OMPSolver<float> solvercr(patchesn, Dtcr, 2, eps, lambda, -1);
    cv::Mat image_result;
    cv::namedWindow("WebCam");
    cv::namedWindow("Diff");
    cv::namedWindow("Gerada");
    // cap.set(cv::CAP_PROP_POS_FRAMES, 15000);
    cv::Mat res;
    cv::Mat3f anterior;
    cv::Mat3f resanterior;
    cv::Mat3f diffimage;
    anterior.copySize(image);
    anterior.setTo(0);
    resanterior.copySize(image);
    resanterior.setTo(cv::Vec3f({0,0,0}));
    for(;;){
        // printval("Image size", ": ", image.size, "");
        cv::imshow("WebCam", image);
        

        // getchar();

        auto tic = std::chrono::system_clock::now();
        cv::Mat img;
        cv::cvtColor(image, res, cv::COLOR_BGR2YCrCb);
        res.convertTo(img, CV_32FC3, 1/255.);
        // img.copyTo(diffimage);
        diffimage = img - resanterior;
        img.copyTo(anterior);

        int col = 0;
        for (int i=0; i<diffimage.rows-sline+1; i+=sline){
            for (int j=0; j<diffimage.cols-scol+1; j+=scol){
                int c=0, a=0, b=0;
                Vector<float> vecy;
                Vector<float> veccb;
                Vector<float> veccr;
                imy.refCol(col, vecy);
                imcb.refCol(col, veccb);
                imcr.refCol(col, veccr);
                while(c<patchesm){
                    auto refmat = diffimage.at<cv::Vec3f>(i + a, j + b);
                    vecy[c] = refmat[0]; //c++;
                    veccb[c] = refmat[2]; //c++;
                    veccr[c] = refmat[1]; c++;
                    a = ++b == scol ? b=0, a+1 : a;
                }
                col++;
            }
        }
        

        SpMatrix<float> rety;
        SpMatrix<float> retcb;
        SpMatrix<float> retcr;
        solvery.solve(imy);
        solvercb.solve(imcb);
        solvercr.solve(imcr);

        auto tac = std::chrono::system_clock::now();
        auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
        // printf("Elapsed encode: %ldms\n", tictac);
        printval("Encode Time", ": ", tictac, "ms");
        
        tic = std::chrono::system_clock::now();
        solvery.getResults(rety);
        solvercb.getResults(retcb);
        solvercr.getResults(retcr);
        Matrix<float> resulty;
        Matrix<float> resultcb;
        Matrix<float> resultcr;
        Dty.mult(rety, resulty);
        Dtcb.mult(retcb, resultcb);
        Dtcr.mult(retcr, resultcr);

        // -- Salvar resultado
        diffimage.setTo(0);
        col = 0;
        for (int i=0; i<diffimage.rows-sline+1; i+=sline){
            // auto imgrow = diffimage.rowRange(i, i+sline);
            for (int j=0; j<diffimage.cols-scol+1; j+=scol){
                // auto refmat = imgrow.colRange(j, j+scol).ptr<float>(0);
                int c=0;
                Vector<float> vecy;
                Vector<float> veccb;
                Vector<float> veccr;
                resulty.refCol(col, vecy);
                resultcb.refCol(col, veccb);
                resultcr.refCol(col, veccr);
                while(c<patchesm){
                    diffimage.at<cv::Vec3f>(i + c/(scol), j + (c)%sline) = cv::Vec3f({vecy[c], veccr[c], veccb[c]});
                    c++;
                }
                col++;
            }
        }

        resanterior += diffimage;
        resanterior.convertTo(image_result, CV_8UC3, 255);
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
        // cv::imshow("Diff", image-anterior);
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