
#include <string.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <windows.h>
#include <fstream>      // std::ifstream
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>


using namespace cv;
using namespace std;

int main( int argc, char** argv ) {


	Mat initial_img;

	string train_img_dir, 
		train_results_file_path,
		test_img_dir;
	
	if (argc != 4) {

		cout << "Training images directory path: ";
		cin >> train_img_dir;
		cout << endl;

		cout << "Training results file path: ";
		cin >> train_results_file_path;
		cout << endl;

		cout << "Test images directory path: ";
		cin >> test_img_dir;
		cout << endl;

	} else {
		train_img_dir = argv[1];
		train_results_file_path = argv[2];
		test_img_dir = argv[3];
	}

	cout << "File path selected: " << train_img_dir << endl;

	
	// detecting keypoints
	SurfFeatureDetector detector(1000);
	vector<KeyPoint> keypoints;	
	
	// computing descriptors
	Ptr<DescriptorExtractor > extractor(new SurfDescriptorExtractor());//  extractor;
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;

	cout << "------- build vocabulary ---------\n";

	cout << "extract descriptors.."<<endl;
	int count = 0;

	HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((train_img_dir + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE) {
		cout << "No files found." << endl;
		return 0;
	}

	do {
		const string file_name = file_data.cFileName;
		const string full_file_name = train_img_dir + "/" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		img = imread(full_file_name);
		detector.detect(img, keypoints);
		extractor->compute(img, keypoints, descriptors);

		training_descriptors.push_back(descriptors);
		cout << ".";


	} while (FindNextFile(dir, &file_data));

	cout << endl;

	cout << "Total descriptors: " << training_descriptors.rows << endl;

	BOWKMeansTrainer bowtrainer(150); //num clusters
	bowtrainer.add(training_descriptors);
	cout << "cluster BOW features" << endl;
	Mat vocabulary = bowtrainer.cluster();

	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);

	//setup training data for classifiers
	map<string,Mat> classes_training_data; classes_training_data.clear();

	cout << "------- train SVMs ---------\n";

	Mat response_hist;
	cout << "look in train data"<<endl;
	count = 0;
	char buf[255];
	ifstream ifs(train_results_file_path);
	int total_samples = 0;
	do
	{
		ifs.getline(buf, 255);
		string line(buf);
		cout << line << endl;
		istringstream iss(line);
		string id, label;
		getline(iss, id, ',');
		getline(iss, label);
		
		img = imread(train_img_dir + "\\" + id);

		bowide.compute(img, keypoints, response_hist);
		
		if(classes_training_data.count(label) == 0) {
			classes_training_data[label].create(0,response_hist.cols,response_hist.type());
		}
		classes_training_data[label].push_back(response_hist);
		total_samples++;

	} while (!ifs.eof());
	cout << endl;

	//train 1-vs-all SVMs
	map<string, Ptr<CvSVM>> classes_classifiers;
	for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
		string class_ = (*it).first;
		cout << "training class: " << class_ << ".." << endl;
		
		Mat samples(0,response_hist.cols,response_hist.type());
		Mat labels(0,1,CV_32FC1);
		
		//copy class samples and label
		samples.push_back(classes_training_data[class_]);
		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
		labels.push_back(class_label);

		//copy rest samples and label
		for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
			string not_class_ = (*it1).first;
			if(not_class_[0] == class_[0]) continue;
			samples.push_back(classes_training_data[not_class_]);
			class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
			labels.push_back(class_label);
		}
		
		Mat samples_32f; samples.convertTo(samples_32f, CV_32F);

		Ptr<CvSVM> tmp_svm = new CvSVM();
		tmp_svm->train(samples_32f,labels);
		classes_classifiers.insert(make_pair(class_, tmp_svm));
	}
	
	cout << "------- test ---------\n";

	//HANDLE dir;
    //WIN32_FIND_DATA file_data;

	/*
    if ((dir = FindFirstFile((test_img_dir + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE) {
		cout << "No files found." << endl;
		return 0;
	}

	do {
		const string file_name = file_data.cFileName;
		const string full_file_name = test_img_dir + "/" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		img = imread(full_file_name);
		bowide.compute(img, keypoints, response_hist);
		//test vs. SVMs
		for (map<string, Ptr<CvSVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second->predict(response_hist,false);
			cout << "class: " << (*it).first << ", response: " << res << endl;
		}


	} while (FindNextFile(dir, &file_data));
	*/


	string ans;
	while (true) {
		cout << "File to test(q to quit): ";
		cin >> ans;
		if (ans.compare("q") == 0) break;
		img = imread(test_img_dir + "/" + ans + ".png");
		bowide.compute(img, keypoints, response_hist);
		for (map<string, Ptr<CvSVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second->predict(response_hist,false);
			cout << "class: " << (*it).first << ", response: " << res << endl;
		}
	} 

	
	cout << endl;
	cout <<"Done."<<endl;


	system("PAUSE");
	return 0;
}