#include "../funclib.hpp"
#include "../NN.hpp"

map<char,int> char2int;
map<int,char> int2char;
char flag;

void ReadFiles(const string &filename,vector<Mat> &output,vector<int> &classId,int &n,int &m){
    n=0; m=0; char p;
    ifstream ifs(filename);
    string path;
    Mat image;
    ifs>>path;
    while(ifs){
        image=imread("../pic/"+path);
        output.push_back(image);
        n++;
        p=path[path.size()-5];
        if(char2int[p]==0){
            if(flag=='\0'){
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
        ifs>>path;
    }
    ifs.close();
}

int main(){

    vector<Mat> trainingset,validationset;
    vector<int> trainingsetClassId,validationsetClassId;
    int trainingsetSize,trainingsetClass;
    int validationsetSize,validationsetClass;

    flag='\0';
    ReadFiles("../trainingset.txt",trainingset,trainingsetClassId,trainingsetSize,trainingsetClass);
    ReadFiles("../validationset.txt",validationset,validationsetClassId,validationsetSize,validationsetClass);

    assert(0==validationsetClass);
    // ensure all letters in the validation set appears in the training set

    Mat training,validation;
    loadMat(trainingset,training);
    loadMat(validationset,validation);

    NN neuralnet(lettersSize*lettersSize,100,trainingsetClass);

    neuralnet.train(training,trainingsetClassId,validation,validationsetClassId,600);

    neuralnet.saveWeights("../NNWeights.txt");
    saveDict("../dict.txt",int2char);

    return 0;
}
