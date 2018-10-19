/*
 * 
 * 
 * video = '../Videos/BigBuckBunny.avi'
 * dirtrain = '../trainframes/'
 * 
 * def main():
    cap = cv2.VideoCapture(video)
    cnt = 0
    while True:
        while cnt % 24 > 0:
            cap.read()
            cnt+=1
        ret, frame = cap.read()
        cnt+=1
        if not ret: break
        cv2.imsave(dirtrain + 'frame_%d.png' % (cnt/24), frame)
*/

#include <iostream>
#include <linalg.h>
#include <decomp.h>
#include <string>
#include "opencv2/opencv.hpp"

auto video = "../Videos/BigBuckBunny.avi";
// auto video = "../Videos/ed_1024.avi";
auto dirtrain = "../trainframes/frame_";

int main(int argc, char* argv[]){
    auto cap = cv::VideoCapture(video);
    cv::Mat frame, frameant;
    std::string fname;
    for(auto cnt=0;;cnt++){
        // frame.copyTo(frameant);
        if (!cap.read(frame)) break;
        fname = std::string(dirtrain) + std::to_string(cnt) + std::string(".png");
        // cv::imwrite(fname, frame-frameant);
        cv::imwrite(fname, frame);
        for (auto i=0; i<23 + 24*10; i++) {
            cap.read(frame);
        }
    }
    return 0;
}