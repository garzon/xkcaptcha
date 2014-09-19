#ifndef FUNCLIB_HPP
#define FUNCLIB_HPP

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>

#include <qstring.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

typedef function<void (Mat&)> actFuncType;
typedef function<Mat (Mat)> actGradFuncType;
typedef function<void (double&)> iterFuncType;

void iterAll(Mat &a,const iterFuncType &f){
    for(auto b=a.begin<double>();b!=a.end<double>();b++){
        f(*b);
    }
}

actFuncType sigmoid=[](Mat &a){
    assert(a.cols==1);
    iterAll(a,[](double &b){
        b=1/(1+exp(-b));
    });
};

actGradFuncType sigmoid_grad=[](Mat a){
    assert(a.cols==1);
    Mat res=a.clone();
    iterAll(res,[](double &b){
        b=(1 - 1/(1+exp(-b)) )/(1+exp(-b));
    });
    return res;
};

Mat addNums(const Mat &a,double num=1.0){
    Size s=a.size();
    s.width++;
    Mat b(s,CV_32F);
    for(int i=0;i<s.height;i++){
        b.at<double>(i,0)=num;
        for(int j=1;j<s.width;j++){
            b.at<double>(i,j)=a.at<double>(i,j-1);
        }
    }
    return b;
}

inline Mat cutFirstCol(const Mat &a){
    return a(Range(0,a.rows-1),Range(1,a.cols-1));
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
        input2hidden(_hiddenDim,1+_inputDim,CV_32F),
        hidden2output(_OutputDim,1+_hiddenDim,CV_32F)
    {
        randu(input2hidden,-epsilon,epsilon);
        randu(hidden2output,-epsilon,epsilon);
    }

    Mat predict(const Mat& input){
        hidden=input*input2hidden.t();
        actFunc(hidden);
        output=hidden*hidden2output.t();
        actFunc(output);
        return output;
    }

    void setTrainingData(const Mat& _input,const vector<int> &_classId){
        assert(_classId.size()==_input.rows);
        input=_input;
        classId=_classId;
    }

    void train(int trainingTimes=100,double learningRate=0.1,int lambda=1){   // a row of the input Mat is an image

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

            input2hidden_grad=Mat::zeros(Size(hiddenDim,inputDim+1),CV_32F);
            hidden2output_grad=Mat::zeros(Size(OutputDim,hiddenDim+1),CV_32F);

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

            J=J/n;

            // TODO: calculate J

            input2hidden=input2hidden-learningRate*input2hidden_grad;
            hidden2output=hidden2output-learningRate*hidden2output_grad;

            cout<<J<<endl;

        }

    }

};


#endif // FUNCLIB_HPP
