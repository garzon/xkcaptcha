#include "funclib.hpp"

int main()
{
    Mat a=imread("/home/garzon/image.jpg"),b;
    b=a.clone();

    cvtColor(a,a,CV_BGR2GRAY);
    threshold(a,a,128,255,THRESH_BINARY);

    trim(a);

    vector<Mat> letters=segment(a);
    Mat input_data(4,901,CV_64F);
    LoadData(input_data,letters);

    NN neuralnet("./weights.txt"); Mat test;
    vector<int> result;
    test=neuralnet.predict(input_data);
    neuralnet.classify(test,result);

    map<int,char> dict;
    loadDict("dict.txt",dict);

    for(auto a:result)
        cout<<dict[a];
    cout<<endl;

    imshow("captcha",b);
    waitKey();

    return 0;
}

