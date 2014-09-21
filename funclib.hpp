#ifndef FUNCLIB_HPP
#define FUNCLIB_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <map>

#include <qstring.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

const int lettersNum=4;
const int lettersSize=30;

void trim(Mat &a){

    int tlx=999999,tly=999999,brx=0,bry=0,x=0,y=0,cols=a.cols;

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

vector<Mat> segment(Mat a,int areaNum,Size areaSize){

    double w=a.cols*1.0/areaNum,h=a.rows;

    vector<Rect> rects;
    double xx=0;
    for(int i=1;i<areaNum;i++){
        rects.emplace(rects.end(),int(xx),0,int(w)+1,h);
        xx+=w+1;
    }
    rects.emplace(rects.end(),int(xx),0,a.cols-int(xx),h);

    vector<Mat> res;
    for(const auto &b: rects){
        res.push_back(a(b));
    }
    for(auto &b: res){
        trim(b);
        resize(b,b,areaSize);
    }
    return res;
}

template <typename T>
void iterAll(Mat &a,const function<void (T&)> &f){
    for(auto b=a.begin<T>();b!=a.end<T>();b++){
        f(*b);
    }
}

void preprocessing(Mat &captcha,int areaNum,Size areaSize,vector<Mat> &letters){
    cvtColor(captcha,captcha,CV_BGR2GRAY);
    threshold(captcha,captcha,128,255,THRESH_BINARY);
    trim(captcha);
    letters=segment(captcha,areaNum,areaSize);
}

void loadMat(vector<Mat> &inputImages,Mat &outputMat){
    outputMat.create(inputImages.size(),inputImages[0].rows*inputImages[0].cols+1,CV_64F);
    int i,j;
    double *p=outputMat.ptr<double>();
    for(i=0;i<inputImages.size();i++){
        *p=1.0;
        p++;
        iterAll<uchar>(inputImages[i],[&](uchar &a){
            *p=a/256.0;
            p++;
        });
    }
}


Mat addNums(const Mat &a,double num=1.0){
    Size s=a.size();
    s.width++;
    Mat b(s,CV_64F);
    for(int i=0;i<s.height;i++){
        b.at<double>(i,0)=num;
        for(int j=1;j<s.width;j++){
            b.at<double>(i,j)=a.at<double>(i,j-1);
        }
    }
    return b;
}

inline Mat cutFirstCol(const Mat &a){
    return a(Range(0,a.rows),Range(1,a.cols));
}

void saveDict(const string &filename,const map<int,char> &dict){
    ofstream ofs(filename);
    for(auto i=dict.begin();i!=dict.end();i++){
        ofs<<i->first<<" "<<i->second<<endl;
    }
    ofs.close();
}

void loadDict(const string &filename,map<int,char> &dict){
    ifstream ifs(filename);
    int a; char b;
    while(ifs){
        ifs>>a>>b;
        dict[a]=b;
    }
    ifs.close();
}

double vectorCompare(const vector<int> &a,const vector<int> &b){
    int n=a.size(),m=0;
    assert(n==b.size());
    for(int i=0;i<n;i++){
        if(a[i]==b[i]) m++;
    }
    double res=m;
    res=res/n;
    return res;
}

inline int double2percent(double a){
    int res;
    res=int(round(a*100));
    return res;
}

#endif // FUNCLIB_HPP
