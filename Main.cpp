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
		test_img_dir,svm_path="null",voc_path="null";
	
	if (argc < 4) {

		cout << endl;
		cout << "Training images directory path: ";
		cin >> train_img_dir;
		cout << endl;

		cout << "Training images solution file path: ";
		cin >> train_results_file_path;
		cout << endl;
		
		cout << "Test images directory path: ";
		cin >> test_img_dir;
		cout << endl;

		cout << "SVM directory path (.yml or .xml): ";
		cin >> svm_path;
		cout << endl;

		cout << "Vocabulary directory path (.yml or .xml): ";
		cin >> voc_path;
		cout << endl;

	} else {
		train_img_dir = argv[1];
		train_results_file_path = argv[2];
		test_img_dir = argv[3];
		if(argc>=5){
			svm_path = argv[4];
		}
		else {
			cout << "SVM directory path (.yml or .xml): ";
			cin >> svm_path;
			cout << endl;
		}
		if(argc>=6){
			voc_path = argv[5];
		}
		else {
			cout << "Vocabulary directory path (.yml or .xml): ";
			cin >> voc_path;
			cout << endl;
		}
	}

	cout << endl;
	cout << "Training images directory path selected: " << train_img_dir << endl;
	cout << "Training images solution file path selected: " << train_results_file_path << endl;
	cout << "Test images directory path selected: " << test_img_dir << endl;
	cout << "SVM path selected: " << svm_path << endl;
	cout << "Vocabulary path selected: " << voc_path << endl;
	cout << endl;

	clock_t begin = clock();
	
	// detecting keypoints
	/*
	SurfFeatureDetector detector; // very bad
	SiftFeatureDetector detector2;
	OrbFeatureDetector detector3; // very bad
	FastFeatureDetector detector4;
	DenseFeatureDetector detector5; // too much bad points
	*/
	SiftFeatureDetector detector;
	FastFeatureDetector detector2;
	vector<KeyPoint> keypoints;	
	
	// computing descriptors
	Ptr<DescriptorExtractor > extractor(new OpponentColorDescriptorExtractor(Ptr<DescriptorExtractor>(new SiftDescriptorExtractor())));
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;	
	
	cout << "Using\n" << typeid(detector).name() << endl << typeid((extractor)).name() << endl << endl;

	HANDLE dir;
	WIN32_FIND_DATA file_data;

	Mat vocabulary;
	FileStorage fs( voc_path, FileStorage::READ );
	if( fs.isOpened()) {
        fs["vocabulary"] >> vocabulary;
		cout << "Vocabulary read! \n";
    }
    else {
        fs.release();
        // Compute your vocabulary.

		cout << "Building vocabulary..." << endl;
		int count = 0;

		if ((dir = FindFirstFile((train_img_dir + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE) {
			cout << "No files found." << endl;
			return 0;
		}

		int empty_keypoints_c = 0;
	
		#pragma omp parallel for schedule(dynamic,3)
		{
			//int counter=0;
			do {
				//if(counter++>5000) break;
				const string file_name = file_data.cFileName;
				const string full_file_name = train_img_dir + "/" + file_name;
				const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

				if (file_name[0] == '.')
					continue;

				if (is_directory)
					continue;

				img = imread(full_file_name);
				Mat img_g;
				cvtColor(img, img_g, CV_RGB2GRAY);
				GaussianBlur( img_g, img_g, Size( 1, 1 ), 0, 0 );
		
				detector.detect(img_g, keypoints);

				/*int detectors_vals[5];
				string detectors[5] = {"SURF","SIFT","ORB","FAST","DENSE"};
				//-- Draw keypoints
				Mat ik1,ik2,ik3;
				//detector.detect(img_g, keypoints);
				//detectors_vals[0]=keypoints.size();
				//cout << "SURF: " << keypoints.size() << endl;
				detector2.detect(img_g, keypoints);
				detectors_vals[1]=keypoints.size();
				cout << "SIFT: " << keypoints.size() << endl;
				drawKeypoints( img_g, keypoints, ik1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
				//detector3.detect(img_g, keypoints);
				//detectors_vals[2]=keypoints.size();
				//cout << "ORB: " << keypoints.size() << endl;
				detector4.detect(img_g, keypoints);
				detectors_vals[3]=keypoints.size();
				cout << "FAST: " << keypoints.size() << endl;
				drawKeypoints( img_g, keypoints, ik2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
				detector5.detect(img_g, keypoints);
				detectors_vals[4]=keypoints.size();
				cout << "DENSE: " << keypoints.size() << endl;
				drawKeypoints( img_g, keypoints, ik3, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
				cout << "Best: " << detectors[distance(detectors_vals, max_element(detectors_vals,detectors_vals+5))] << endl << endl;
				/*-- Show detected (drawn) keypoints
				imshow("Sample", img);
				imshow("SIFT", ik1 );
				imshow("FAST", ik2 );
				imshow("DENSE", ik3 );
				waitKey();/**/

				if(keypoints.empty()) {
					
					detector2.detect(img_g, keypoints);
					if(keypoints.empty()) {
						#pragma omp critical
						empty_keypoints_c++;
						//cout << "Unable to detect any keypoints for this image." << endl;
						continue;
					}
				}

				extractor->compute(img, keypoints, descriptors);
				#pragma omp critical
				{
					training_descriptors.push_back(descriptors);
				}

				cout << "Number of empty keypoints: " << empty_keypoints_c << "  Total descriptors: " << training_descriptors.rows << "\r" ;

			} while (FindNextFile(dir, &file_data));
		}

		cout << endl << endl;

		BOWKMeansTrainer bowtrainer(1000); //num clusters
		bowtrainer.add(training_descriptors);
		cout << "Clustering Bag of Words features..." << endl;
		vocabulary = bowtrainer.cluster();

        fs.open( voc_path, FileStorage::WRITE );
        fs << "vocabulary" << vocabulary; // TODO 
    }
    fs.release();
		
	cout << endl << "Reading SVM.." << endl;
	
	Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
	//BOWImgDescriptorExtractor bowide(extractor,matcher);
	//bowide.setVocabulary(vocabulary);
	Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(extractor,matcher));
	Mat response_hist;
	string class_labels[10] = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	CvSVM svm;
	svm.load(svm_path.c_str());
    if( !svm.get_support_vector_count() > 0 ) // model not loaded
    {
		cout << "SVM Failed to read.." << endl;
		bowide->setVocabulary(vocabulary);


		//setup training data for classifiers
		map<string,Mat> classes_training_data;
		//vector<string> class_labels;
		classes_training_data.clear();

		cout << "Training SVMs..." << endl;

		int count = 0;
		char buf[255];
		ifstream ifs(train_results_file_path);

		#pragma omp parallel for schedule(dynamic,3)
		{
			do {
				ifs.getline(buf, 255);
				string line(buf);
				istringstream iss(line);
				string id, label;
				getline(iss, id, ',');
				getline(iss, label);
		
				string imgpath = train_img_dir + "\\" + id + ".png";
				img = imread(imgpath);
				cout << '\r' << "Loading: " << imgpath;

				keypoints.clear();
				detector.detect(img, keypoints);
				//bowide.compute(img, keypoints, response_hist);
				bowide->compute(img, keypoints, response_hist);
		
				#pragma omp critical
				{
					if(classes_training_data.count(label) == 0) {
						classes_training_data[label].create(0,response_hist.cols,response_hist.type());
						//class_labels.push_back(label);
					}
					classes_training_data[label].push_back(response_hist);
				}

			} while (!ifs.eof());
		}

		cout << '\r' << "                                           " << endl;


		//train 1-vs-all SVMs
		CvSVM svm;
		CvSVMParams params = CvSVMParams(); params.svm_type = CvSVM::C_SVC; params.kernel_type = CvSVM::LINEAR;
		map<string, Ptr<CvSVM>> classes_classifiers;
		int num = 1;
	
		Mat samples(0,response_hist.cols,response_hist.type());
		Mat labels(0,1,CV_32FC1);
		Mat samples_32f; 

		for (int i=0;i<10/*class_labels.size()*/;i++) {
			string label = class_labels[i];
			if(classes_training_data.count(label) == 0) {
				continue;
			}
			Scalar val = Scalar(i,i,i);
			samples.push_back(classes_training_data[label]);
			Mat class_label = Mat(classes_training_data[label].rows, 1, CV_32FC1, val);
			labels.push_back(class_label);
		}

		samples.convertTo(samples_32f, CV_32FC1);
		//svm.train(samples_32f,labels,Mat(),Mat(),params);
		svm.train(samples_32f,labels);

		svm.save(svm_path.c_str()); // TODO
	
	}

	//cout << "Training classes..." << endl;

	////train 1-vs-all SVMs
	//map<string, Ptr<CvSVM>> classes_classifiers;
	//int num = 1;

	//for (map<string,Mat>::iterator it = classes_training_data.begin(); it != classes_training_data.end(); ++it) {
	//	string class_ = (*it).first;
	//	cout << "#" << num++ << " " << class_ << "..." << endl;
	//	
	//	Mat samples(0,response_hist.cols,response_hist.type());
	//	Mat labels(0,1,CV_32FC1);
	//	
	//	//copy class samples and label
	//	samples.push_back(classes_training_data[class_]);
	//	Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
	//	labels.push_back(class_label);

	//	//copy rest samples and label
	//	for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
	//		string not_class_ = (*it1).first;
	//		if(not_class_[0] == class_[0]) continue;
	//		samples.push_back(classes_training_data[not_class_]);
	//		class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
	//		labels.push_back(class_label);
	//	}
	//	
	//	Mat samples_32f; 
	//	samples.convertTo(samples_32f, CV_32F);

	//	/*
	//	Ptr<CvSVM> tmp_svm = new CvSVM();
	//	CvSVMParams params = CvSVMParams();
	//	params.svm_type    = CvSVM::C_SVC;
	//	auto lastInsertPtr = classes_classifiers.insert(make_pair(class_, tmp_svm));
	//	lastInsertPtr.first->second->train(samples_32f,labels,Mat(),Mat(),params);
	//	*/
	//	
	//	Ptr<CvSVM> tmp_svm = new CvSVM();
	//	auto lastInsertPtr = classes_classifiers.insert(make_pair(class_, tmp_svm));
	//	lastInsertPtr.first->second->train(samples_32f,labels);
	//	
	//}

	cout << endl;

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time elasped building vocabulary and training classes: " << elapsed_secs << " seconds." << endl << endl;
	
	string ans;
	cout << "Please select the testing mode ([1]Manual [2]Directory): ";
	cin >> ans;
	if (ans.compare("1") == 0) { // MANUAL MODE
		cout << endl << "Manual mode selected. Testing..." << endl;

		while (true) {
			cout << "File to test ([q] to quit): ";
			cin >> ans;
			if (ans.compare("q") == 0) break;
			img = imread(test_img_dir + "/" + ans + ".png");
			Mat img_g;
			cvtColor(img, img_g, CV_RGB2GRAY);
			GaussianBlur( img_g, img_g, Size( 1, 1 ), 0, 0 );

			keypoints.clear();
			detector.detect(img_g, keypoints);

			if(keypoints.empty()) {
				cout << "Unable to detect any keypoints for this image." << endl;
				continue;
			}
			
			//bowide.compute(img, keypoints, response_hist);
			bowide->compute(img, keypoints, response_hist);

			/*for (map<string, Ptr<CvSVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
				int res = (*it).second->predict(response_hist,false);
				cout << "class: " << (*it).first << ", response: " << res << endl;
			}*/
			Mat sample;
			Mat samples(0,response_hist.cols,response_hist.type());
			samples.push_back(response_hist);
			response_hist.convertTo(samples, CV_32FC1);
			cout << "Rows in the sample: " << sample.rows << " : " << sample.cols << endl << "Rows in the histogram: " << response_hist.rows << " : " << response_hist.cols << endl;
			int res = svm.predict(samples, false);
			cout << "class: " << class_labels[res];
		} 

	} else { //DIRECTORY MODE
		cout << endl << "Directory mode selected. Testing..." << endl;

		if ((dir = FindFirstFile((test_img_dir + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE) {
			cout << "No files found." << endl;
			return 0;
		}

		ofstream outputFile;
		outputFile.open("submission.csv", ios::out | ios::trunc);
		outputFile << "id,label\n";

		
		#pragma omp parallel for schedule(dynamic,3)
		{
			do {
				const string file_name = file_data.cFileName;
				const string full_file_name = test_img_dir + "/" + file_name;
				const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

				if (file_name[0] == '.')
					continue;

				if (is_directory)
					continue;

				img = imread(full_file_name);
				Mat img_g;
				cvtColor(img, img_g, CV_RGB2GRAY);
				GaussianBlur( img_g, img_g, Size( 1, 1 ), 0, 0 );
				keypoints.clear();
				detector.detect(img_g, keypoints);
				if(keypoints.empty()) {
					//cout << "Unable to detect any keypoints for this image." << endl;
					// sem saber poe sempre 0 :)
					#pragma omp critical
					outputFile << file_name << ",0" << endl;
					continue;
				}
				//bowide.compute(img, keypoints, response_hist);
				bowide->compute(img, keypoints, response_hist);
		
				/*for (map<string, Ptr<CvSVM>>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
					int res = (*it).second->predict(response_hist,false);
					cout << "class: " << (*it).first << ", response: " << res << endl;
				}*/
				int res = svm.predict(response_hist);
				#pragma omp critical
				outputFile << file_name << "," << class_labels[res] << endl;

			} while (FindNextFile(dir, &file_data));
		}

		outputFile.close();
	}
	
	cout << endl << endl;

	clock_t end2 = clock();
	elapsed_secs = double(end2 - end) / CLOCKS_PER_SEC;

	cout << "Time elasped testing: " << elapsed_secs << " seconds." << endl << endl;

	system("PAUSE");
	
	return 0;
}