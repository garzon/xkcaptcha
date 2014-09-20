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

typedef function<void (Mat&)> actFuncType;
typedef function<Mat (Mat)> actGradFuncType;

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

template <typename T>
void iterAll(Mat &a,const function<void (T&)> &f){
    for(auto b=a.begin<T>();b!=a.end<T>();b++){
        f(*b);
    }
}

void LoadData(Mat &outputMat,vector<Mat> &inputVec){
    int i,j;
    double *p=outputMat.ptr<double>();
    for(i=0;i<inputVec.size();i++){
        *p=1.0;
        p++;
        iterAll<uchar>(inputVec[i],[&](uchar &a){
            *p=a/256.0;
            p++;
        });
        //cout<<outputMat.row(i)<<endl;
    }
}

actFuncType sigmoid=[](Mat &a){
    iterAll<double>(a,[](double &b){
        b=1/(1+exp(-b));
    });
};

actGradFuncType sigmoid_grad=[](Mat a){
    Mat res=a.clone();
    iterAll<double>(res,[](double &b){
        b=(1 - 1/(1+exp(-b)) )/(1+exp(-b));
    });
    return res;
};

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

class NN{
    int inputDim,hiddenDim,OutputDim;
    // Vectors:
    Mat hidden,output;
    // Weight Mats:
    Mat input2hidden,hidden2output;
    // Accumulate the grad
    Mat input2hidden_grad,hidden2output_grad;
    const actFuncType & actFunc;
    const actGradFuncType & actGradFunc;

    const double epsilon{0.12};

    // storage
    Mat input;
    vector<int> classId;

public:
    NN(int _inputDim,int _hiddenDim,int _OutputDim,const actFuncType &_actFunc=sigmoid,const actGradFuncType &_actGradFunc=sigmoid_grad):
        inputDim(_inputDim),
        hiddenDim(_hiddenDim),
        OutputDim(_OutputDim),
        actFunc(_actFunc),
        actGradFunc(_actGradFunc),
        input2hidden(_hiddenDim,1+_inputDim,CV_64F),
        hidden2output(_OutputDim,1+_hiddenDim,CV_64F)
    {
        randu(input2hidden,-epsilon,epsilon);
        randu(hidden2output,-epsilon,epsilon);
    }

    NN(const string &filename,const actFuncType &_actFunc=sigmoid,const actGradFuncType &_actGradFunc=sigmoid_grad):
        actFunc(_actFunc),
        actGradFunc(_actGradFunc)
    {
        ifstream ifs(filename);
        ifs>>inputDim>>hiddenDim>>OutputDim;
        input2hidden.create(hiddenDim,1+inputDim,CV_64F);
        hidden2output.create(OutputDim,1+hiddenDim,CV_64F);
        iterAll<double>(input2hidden,[&](double &a){
           ifs>>a;
        });
        iterAll<double>(hidden2output,[&](double &a){
           ifs>>a;
        });
        ifs.close();
    }

    void saveWeights(const string &filename){
        ofstream ofs(filename);
        ofs<<inputDim<<" "<<hiddenDim<<" "<<OutputDim<<endl;
        iterAll<double>(input2hidden,[&](double &a){
            ofs<<a<<" ";
        });
        ofs<<endl<<endl;
        iterAll<double>(hidden2output,[&](double &a){
           ofs<<a<<" ";
        });
        ofs<<endl;
        ofs.close();
    }
    Mat predict(const Mat& input){
        hidden=input*input2hidden.t();
        actFunc(hidden);
        output=addNums(hidden)*hidden2output.t();
        actFunc(output);
        return output;
    }
    void classify(const Mat &predict_output,vector<int> &result){
        result.clear();
        const double *p; int imax=0; double maxval; int n=predict_output.rows;
        for(int i=0;i<n;i++){
            p=predict_output.ptr<double>(i);
            maxval=-1;
            for(int j=0;j<OutputDim;j++){
                if(*p>maxval){
                    maxval=*p;
                    imax=j;
                }
                p++;
            }
            result.push_back(imax);
        }
    }
    void setTrainingData(const Mat& _input,const vector<int> &_classId){
        assert(_classId.size()==_input.rows);
        input=_input;
        classId=_classId;
    }

    void train(int trainingTimes=300,double learningRate=0.1,int lambda=1){   // a row of the input Mat is an image

        int n=input.rows;

        for(int zzz=0;zzz<trainingTimes;zzz++){

            predict(input);

            Mat delta_Output=output.clone();
            for(int i=0;i<n;i++)
                delta_Output.at<double>(i,classId[i])=delta_Output.at<double>(i,classId[i])-1;

            Mat delta_hidden=delta_Output*hidden2output;
            delta_hidden=cutFirstCol(delta_hidden);
            Mat tmp=actGradFunc(input*input2hidden.t());
            multiply(delta_hidden,tmp,delta_hidden);

            input2hidden_grad=Mat::zeros(hiddenDim,inputDim+1,CV_64F);
            hidden2output_grad=Mat::zeros(OutputDim,hiddenDim+1,CV_64F);

            for(int i=0;i<n;i++){
                hidden2output_grad=hidden2output_grad+delta_Output.row(i).t()*addNums(hidden.row(i));
                input2hidden_grad=input2hidden_grad+delta_hidden.row(i).t()*input.row(i);
            }

            input2hidden_grad=(input2hidden_grad+addNums(cutFirstCol(input2hidden),0)*lambda)/n;
            hidden2output_grad=(hidden2output_grad+addNums(cutFirstCol(hidden2output),0)*lambda)/n;

            double J=0;

            for(int i=0;i<n;i++){
                for(int j=0;j<OutputDim;j++){
                    if(j!=classId[i])
                        J+=log(1-output.at<double>(i,j));
                    else
                        J+=log(output.at<double>(i,j));
                }
            }

            J=-J/n;

            // TODO: calculate J

            input2hidden=input2hidden-learningRate*input2hidden_grad;
            hidden2output=hidden2output-learningRate*hidden2output_grad;

            cout<<J<<endl;

        }

    }

};


#endif // FUNCLIB_HPP
