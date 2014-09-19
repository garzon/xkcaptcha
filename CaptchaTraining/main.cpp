#include <fstream>
#include <map>

#include "funclib.hpp"

map<char,int> char2int;
map<int,char> int2char;

void ReadFiles(vector<Mat> &output,vector<int> &classId,int &n){
    n=0; int m=0; char p;
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
    int n;
    ReadFiles(input,classId,n);
    /*int i=input.size()-1;
    //for(int i=0;i<50;i++){
        cout<<classId[i]<<" "<<int2char[classId[i]]<<endl;
        imshow("output",input[i]);
        waitKey();
    //}
    */
    return 0;
}
