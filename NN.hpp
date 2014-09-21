#ifndef NN_HPP
#define NN_HPP

#include "funclib.hpp"

class NN{
public:

    typedef function<Mat (Mat)> actFuncType;

    class actFunc{
    public:
        static const actFuncType sigmoid;
        static const actFuncType sigmoid_grad;
    };

private:

    int inputDim,hiddenDim,OutputDim;
    // Vectors:
    Mat hidden,output;
    // Weight Mats:
    Mat input2hidden,hidden2output;
    // Accumulate the grad
    Mat input2hidden_grad,hidden2output_grad;
    const actFuncType & actFunc;
    const actFuncType & actFuncGrad;

    const double epsilon{0.12};

    // storage

public:

    // construct a new NN
    NN(int _inputDim,int _hiddenDim,int _OutputDim,const actFuncType &_actFunc=actFunc::sigmoid,const actFuncType &_actFuncGrad=actFunc::sigmoid_grad):
        inputDim(_inputDim),
        hiddenDim(_hiddenDim),
        OutputDim(_OutputDim),
        actFunc(_actFunc),
        actFuncGrad(_actFuncGrad),
        input2hidden(_hiddenDim,1+_inputDim,CV_64F),
        hidden2output(_OutputDim,1+_hiddenDim,CV_64F)
    {
        randu(input2hidden,-epsilon,epsilon);
        randu(hidden2output,-epsilon,epsilon);
    }

    // construct NN from saved weights in the file
    NN(const string &filename,const actFuncType &_actFunc=actFunc::sigmoid,const actFuncType &_actFuncGrad=actFunc::sigmoid_grad):
        actFunc(_actFunc),
        actFuncGrad(_actFuncGrad)
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

    // save NN weights in the file
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

    // forward
    Mat predict(const Mat& input){
        hidden=input*input2hidden.t();
        output=addNums(actFunc(hidden))*hidden2output.t();
        return actFunc(output);
    }

    // predict
    void classify_prediction(const Mat& predict_output,vector<int> &result){
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

    // wrapper = classify_prediction * predict
    void classify(const Mat& input,vector<int> &result){
        Mat predict_output=predict(input);
        classify_prediction(predict_output,result);
    }

    // train
    void train(const Mat& input,const vector<int> &classId,
               const Mat& validation,const vector<int> &validationClass,
               int trainingTimes=300,double learningRate=0.1,int lambda=1)
    {
        // a row of the input Mat is an image

        int n=input.rows;

        vector<int> result;

        for(int zzz=0;zzz<trainingTimes;zzz++){

            predict(input);

            Mat delta_Output=actFunc(output);
            for(int i=0;i<n;i++)
                delta_Output.at<double>(i,classId[i])=delta_Output.at<double>(i,classId[i])-1;

            Mat delta_hidden=delta_Output*hidden2output;
            delta_hidden=cutFirstCol(delta_hidden);
            Mat tmp=actFuncGrad(hidden);
            multiply(delta_hidden,tmp,delta_hidden);

            input2hidden_grad=Mat::zeros(hiddenDim,inputDim+1,CV_64F);
            hidden2output_grad=Mat::zeros(OutputDim,hiddenDim+1,CV_64F);

            tmp=actFunc(hidden);

            for(int i=0;i<n;i++){
                hidden2output_grad=hidden2output_grad+delta_Output.row(i).t()*addNums(tmp.row(i));
                input2hidden_grad=input2hidden_grad+delta_hidden.row(i).t()*input.row(i);
            }

            input2hidden_grad=(input2hidden_grad+addNums(cutFirstCol(input2hidden),0)*lambda)/n;
            hidden2output_grad=(hidden2output_grad+addNums(cutFirstCol(hidden2output),0)*lambda)/n;

            tmp=actFunc(output);

            double J=0;

            for(int i=0;i<n;i++){
                for(int j=0;j<OutputDim;j++){
                    if(j!=classId[i])
                        J+=log(1-tmp.at<double>(i,j));
                    else
                        J+=log(tmp.at<double>(i,j));
                }
            }

            J=-J/n;

            // TODO: calculate J

            input2hidden=input2hidden-learningRate*input2hidden_grad;
            hidden2output=hidden2output-learningRate*hidden2output_grad;

            classify(validation,result);

            cout<<"iteration "<<zzz<<": J="<<J<<" Validation: "<<double2percent(vectorCompare(result,validationClass))<<"%"<<endl;

        }

    }

};

// some activation functions

const NN::actFuncType NN::actFunc::sigmoid=[](Mat a){
    Mat res=a.clone();
    iterAll<double>(res,[](double &b){
        b=1/(1+exp(-b));
    });
    return res;
};

const NN::actFuncType NN::actFunc::sigmoid_grad=[](Mat a){
    Mat res=a.clone();
    iterAll<double>(res,[](double &b){
        b=(1 - 1/(1+exp(-b)) )/(1+exp(-b));
    });
    return res;
};


#endif // NN_HPP
