#include "funclib.hpp"

int main()
{
    QString path("./pic/%1%2.%3");
    for(int i=0;i>-1;i++){

        Mat a=imread(path.arg(i).arg("").arg("jpg").toStdString());
        if(a.empty()) break;

        cvtColor(a,a,CV_BGR2GRAY);
        threshold(a,a,128,255,THRESH_BINARY);

        trim(a);

        imshow("input",a);

        vector<Mat> letters=segment(a);
        int j=-1;
        for(const auto &letter: letters){
            j++;
            imshow("letter",letter);
            int c=waitKey(0);
            cout<<char(c)<<endl;
            imwrite(path.arg(i*4+j).arg(char(c)).arg("bmp").toStdString(),letter);
        }
        system(("rm "+path.arg(i).arg("").arg("jpg").toStdString()).c_str());
    }
    return 0;
}

