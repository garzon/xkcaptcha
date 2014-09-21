#include "../funclib.hpp"
#include "../NN.hpp"

int main()
{
    Mat a=imread("../image.jpg"),b;
    b=a.clone();

    vector<Mat> letters;
    preprocessing(a,lettersNum,Size(lettersSize,lettersSize),letters);

    Mat input_data;
    loadMat(letters,input_data);

    NN neuralnet("../NNWeights.txt");

    vector<int> result;
    neuralnet.classify(input_data,result);

    map<int,char> dict;
    loadDict("../dict.txt",dict);

    for(auto a:result)
        cout<<dict[a];
    cout<<endl;

    return 0;
}

