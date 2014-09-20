#include <fstream>
#include <map>

#include "funclib.hpp"

map<char,int> char2int;
map<int,char> int2char;

void ReadFiles(vector<Mat> &output,vector<int> &classId,int &n,int &m){
    n=0; m=0; char p;
    ifstream ifs("list.txt");
    string path;
    char flag;
    Mat image;
    while(ifs){
        ifs>>path;
        image=imread("./pic/"+path);
        output.push_back(image);
        n++;
        p=path[path.size()-5];
        if(char2int[p]==0){
            if(m==0){
                m++;
                int2char[0]=p;
                flag=p;
            }
            if(p!=flag){
                char2int[p]=m;
                int2char[m]=p;
                m++;
            }
        }
        classId.push_back(char2int[p]);
    }
    ifs.close();
}

int main(){
    vector<Mat> input; vector<int> classId;
    int n,m;
    ReadFiles(input,classId,n,m);

    Mat input_data(n,901,CV_64F);
    LoadData(input_data,input);

    NN neuralnet(900,100,m);
    neuralnet.setTrainingData(input_data,classId);
    neuralnet.train(500);
    neuralnet.saveWeights("./weights.txt");

    saveDict("dict.txt",int2char);

    return 0;
}
