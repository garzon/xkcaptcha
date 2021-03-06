#include <time.h>

#include "../funclib.hpp"

const double ratio=0.9;

int main()
{

    srand(time(0));

    QString path("../pic/%1%2.%3");
    for(int i=0;i>-1;i++){

        Mat a=imread(path.arg(i).arg("").arg("jpg").toStdString());
        if(a.empty()) break;

        vector<Mat> letters;
        preprocessing(a,lettersNum,Size(lettersSize,lettersSize),letters);

        imshow("input",a);

        for(const auto &letter: letters){
            imshow("letter",letter);
            int c=waitKey(0);
            cout<<char(c)<<endl;
            imwrite(path.arg(rand()).arg(char(c)).arg("bmp").toStdString(),letter);
        }
        system(("rm "+path.arg(i).arg("").arg("jpg").toStdString()).c_str());
    }

    system("ls ../pic > ../piclist.txt");
    ifstream ifs("../piclist.txt");
    string filename;
    vector<string> piclist;
    ifs>>filename;
    while(ifs){
        piclist.push_back(filename);
        ifs>>filename;
    }
    ifs.close();

    ofstream ofs("../trainingset.txt");
    int i;
    for(i=0; (i*1.0/piclist.size() < ratio)&&(i<piclist.size()) ; i++){
        ofs<<piclist[i]<<endl;
    }
    ofs.close();

    ofs.open("../validationset.txt");
    for(;i<piclist.size();i++){
        ofs<<piclist[i]<<endl;
    }
    ofs.close();

    return 0;

}

