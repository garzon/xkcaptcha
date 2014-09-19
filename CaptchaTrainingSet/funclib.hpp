#ifndef FUNCLIB_HPP
#define FUNCLIB_HPP

#include <iostream>
#include <vector>
#include <stdlib.h>

#include <qstring.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


void trim(Mat &a){

    int tlx=9999,tly=9999,brx=0,bry=0,x=0,y=0,cols=a.cols;

    for(auto b=a.begin<uchar>();b!=a.end<uchar>();b++){
        if(*b==0){
            tlx=min(tlx,x);
            tly=min(tly,y);
            brx=max(brx,x);
            bry=max(bry,y);
        }
        x++;
        if(x==cols){
            x=0;
            y++;
        }
    }

    a=a(Range(tly,bry),Range(tlx,brx));
}

vector<Mat> segment(Mat a){

    double w=a.cols/4.0,h=a.rows;

    vector<Rect> rects;
    double xx=0;
    rects.emplace(rects.end(),int(xx),0,int(w)+1,h); xx+=w+1;
    rects.emplace(rects.end(),int(xx),0,int(w)+1,h); xx+=w+1;
    rects.emplace(rects.end(),int(xx),0,int(w)+1,h); xx+=w+1;
    rects.emplace(rects.end(),int(xx),0,a.cols-int(xx),h);

    vector<Mat> res;
    for(const auto &b: rects){
        res.push_back(a(b));
    }
    for(auto &b: res){
        trim(b);
        resize(b,b,Size(30,30));
    }
    return res;
}

#endif // FUNCLIB_HPP
