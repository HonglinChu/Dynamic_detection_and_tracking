

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <iostream>

using namespace cv;
using namespace std;


//�������������������  
bool biggerSort(vector<Point> v1, vector<Point> v2)  
{  
    return contourArea(v1)>contourArea(v2);  
}  
void main()
{
    VideoCapture capture;//��ƵԴ
    Mat image01,image02;
    vector<Rect> target;//�������ο�
    int fastThreshold = 100;//���ֵ��ͨ�������ǲ��Ե������𣿣�
    float bili = 0.8;//��������ͼ��׼����Ե����һ�£���˶�ԭͼ��С0.9�ı�����������⵽��Ŀ��

    //capture.open(0);//������ͷ
    capture.open(0);//����Ƶ
    if (capture.isOpened())//�������򿪣�����û�ж���ͼ����ر����
    {
        capture>>image02;
        if(image02.empty())
        {
            capture.release();
        }
    }        
    imshow("pic", image02);
    waitKey(5);

    Mat image1,image2;   
    Mat temp,image02temp;

    vector<KeyPoint> keyPoint1, keyPoint2;//����ͼ�м�⵽��������

    SurfDescriptorExtractor SurfDescriptor;//SURF���������� 
    Mat imageDesc1,imageDesc2;
    int bAdaptFast=0;                      //�Ƿ��������Ӧ����ֵ,����fast��ֵ��ʱ����Ҫ��̫��֡
    vector<DMatch> matchePoints;           //ƥ���������
	vector<vector<Point>>contours;        //����
	while (1)
    {
        target.clear(); //���ο��ڴ���ͷ�
        vector<Rect>(target).swap(target);//����ط���ʲô��˼
		//����һ֡�����ݱ�������
        image1=image2.clone();
        image01=image02.clone();
        imageDesc1=imageDesc2.clone();
        keyPoint1=keyPoint2;
        //��һ֡ͼ
        int num = 5;  
        if (!bAdaptFast)//û���������Ӧ��ֵ����
        {
            num = 1;
        }
        bool flag = true;
        while (num-- && flag)//���������Ӧ���ھ�ѭ��5�Σ�û���������Ӧ������ѭ��1��
        {
            capture >> image02;//
            //imageSave=image02.clone();//���ݵ�saveͼ��
            if(image02.empty())
            {
                keyPoint1.clear();
                keyPoint2.clear();
                image01.release();
                image02.release();
                image1.release();
                image2.release();
                imageDesc1.release();
                imageDesc2.release();
                capture.release();
                target.clear();  
                temp.release();
                image02temp.release();    
                flag = false;
                exit(0);
                break;
            }
        }
        if (!flag)//�����ȡ����ͼƬΪ�վ�ֱ������ѭ��
        {
            break;
        }           
        //�Ҷ�ͼת��,�ڴ�֮ǰimage2����Ϣ�Ѿ����ݸ���image1
        cvtColor(image02,image2,CV_BGR2GRAY);  
        //��ȡ������   
        FAST(image2, keyPoint2, fastThreshold);//fast��������
        //����ƥ�����������������FAST��ֵ�Ķ�̬����
        //��ⲻ��������ƥ��ᱨ������0�Ļ���û��Ҫ�ڽ���ƥ�䣬��Ȼ�Ǻ����ƥ��ᱨ��
        if (keyPoint2.size() == 0)
        {
            fastThreshold -= 2;  
            bAdaptFast = 0;//û�е�����ֵ
            imshow("pic", image02);
            waitKey(10);
            continue;//�����ⲻ�������㣬�Ͳ���ִ�к����������ƥ�����
        }
        //������������Ϊ�±ߵ�������ƥ����׼��      
        SurfDescriptor.compute(image2, keyPoint2, imageDesc2);
        //���ƥ�������㣬����ȡ�������     
        FlannBasedMatcher matcher;  
        //����ƥ��
        matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());  
        sort(matchePoints.begin(), matchePoints.end());//����������   
        
        if (matchePoints.size()<100)//���������������һ�����ƣ������⵽��������С��100��
        {
            fastThreshold-=1;
            if(fastThreshold <10)
            {
                fastThreshold =10;
            }
            bAdaptFast = 0;//û�е�����ֵ
            imshow("pic", image02);
            waitKey(10);
            continue;//��ֵû�е�����
        }
        if(matchePoints.size()>300)//���������������һ������
        {
            fastThreshold += int(matchePoints.size()/100);//�����ʽ��������ô������������
            bAdaptFast = 0;//û�е�����ֵ
            imshow("pic",image02);
            waitKey(10);
            continue;
        }
        bAdaptFast = 1;//�����ִ�е�����˵���Ѿ�����������ֵ
        //ɸ����ƥ��������
		vector<Point2f> imagePoints1,imagePoints2;  
        for(int i=0; i < matchePoints.size()* 0.5; i++)//ֻѡȡ����ƥ�䵽���������ǰ�ٷ�֮��ʮ 
        {         
            imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);       
            imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);       
        }  
        //��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
        Mat homo = findHomography(imagePoints1,imagePoints2,CV_RANSAC);//CV_RANSAC³���Էǳ��� 
        //ͼ����׼  
        Mat imageTransform1,imgpeizhun,imgerzhi;
		//����һ֡ͼƬͨ��ͶӰӳ�����ת������һ֡������ϵ�¡�������ת���߶Ȳ����Ρ�
        warpPerspective(image01, imageTransform1,homo,Size(image02.cols, image02.rows));    
        //imshow("����͸�Ӿ���任��",imageTransform1); 
		//ͼ����,�õ����ڵ�ǰ֡����ϵ���������ͼ��Ĳ��
        absdiff(image02, imageTransform1, imgpeizhun);//ע������ͼ��ߴ�һ��Ҫһ��
        imshow("��׼diff",imgpeizhun);  
        Mat imgOtsu =imgpeizhun(Rect(imgpeizhun.cols *(1-bili), imgpeizhun.rows*(1-bili),imgpeizhun.cols * (2 * bili - 1), imgpeizhun.rows*(2 * bili - 1)));
        double MaxValue;
        minMaxLoc(imgOtsu, 0, &MaxValue);//�Ҳ��ͼ������ֵ
		//��������������ô�趨�ģ���������������MaxValue*0.7������Զ�̬����ô��������
        threshold(imgpeizhun, imgerzhi, MaxValue*0.7/*- 10*/, 255.0 , CV_THRESH_BINARY);//10Ϊ�������
        //threshold(imgpeizhun, imgerzhi, dlg->GetDlgItemInt(IDC_EDIT_DIFF), 255.0 , CV_THRESH_BINARY);
        //imshow("��׼��ֵ��", imgerzhi);
        image02temp = image02.clone();
        cvtColor(imgerzhi, temp, CV_RGB2GRAY);//ת���ɻҶ� 
        //������ͨ��
        //Mat se=getStructuringElement(MORPH_RECT, Size(3,3));
        //morphologyEx(temp, temp,MORPH_OPEN,se);
        int dialate_size=9;//����size,�����ֵ�Ĵ�С��Ӱ��Ч���ɣ�����˵����3,5,7��8,9
        Mat se2=getStructuringElement(MORPH_RECT,Size(dialate_size,dialate_size));
        morphologyEx(temp,temp,MORPH_DILATE,se2);
        findContours(temp,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);//������е��ⲿ����
        //����������ɸѡ
        if (contours.size() < 1)
        {
            imshow("pic", image02);
            waitKey(10);
            continue;
        }
        sort(contours.begin(), contours.end(), biggerSort);//������С������
        //Ŀ��ȷ��
        if(true)//��ѡģʽ
        {
            int count=(contours.size() > 12)? 12:contours.size();//Ŀ����������ಶ׽12��Ŀ��ô����
            for (int k = 0; k < count; k++)
            {
                Rect bomen = boundingRect(contours[k]);//��������Ӿ���
                //ʡ��������׼�����ı�Ե��Ч��Ϣ
                if (bomen.x > image02temp.cols * (1 - bili) && bomen.y > image02temp.rows * (1 - bili) 
                    && bomen.x + bomen.width < image02temp.cols * bili && bomen.y + bomen.height < image02temp.rows * bili
                    /*&& contourArea(contours[k]) > contourArea(contours[0])/10*/
                    && contourArea(contours[k])>20 && contourArea(contours[k])<1000)//�����������ƿ��Խ��ж�̬����
                {         
                    //ɸѡĿ����״
                    if (bomen.width > bomen.height)
                    {
                        if (bomen.width > 2 * bomen.height)
                        {
                            continue;
                        }
                    }
                    if (bomen.width < bomen.height)
                    {
                        if (2 * bomen.width < bomen.height)
                        {
                            continue;
                        }
                    }
					
                    Rect temp;
                    int offset = 5;//44*44
                    temp.x = bomen.x + bomen.width / 2 - offset;
                    temp.y = bomen.y + bomen.height / 2 - offset;
                    temp.width = offset * 2;
                    temp.height = offset * 2;
                    bomen = temp;

                    rectangle(image02temp, bomen, Scalar(0,0,255), 2, 8, 0);
                    target.push_back(bomen);
                }
            }
        }
        imshow("pic", image02temp);
        waitKey(10);  
    }
}


////////////////////////////////////////��֡���1��6,2��7,3��8��4��9,5��10;6��11,7��12,8��13,9��14,10��15///////////////////////////////////////////
//#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp> 
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//#define  num 5
//int   NUM=5;
////�������������������  
//bool biggerSort(vector<Point> v1, vector<Point> v2)  
//{  
//    return contourArea(v1)>contourArea(v2);  
//}  
//void main()
//{
//    VideoCapture capture;//��ƵԴ
//    Mat image01[num],image02[num];
//    vector<Rect> target;//�������ο�
//    int fastThreshold = 100;//���ֵ��ͨ�������ǲ��Ե�������
//    float bili = 0.9;//��������ͼ��׼����Ե����һ�£���˶�ԭͼ��С0.9�ı�����������⵽��Ŀ��
//    //capture.open(0);//������ͷ
//    capture.open("move.mp4");//����Ƶ
//    if (capture.isOpened())//�������򿪣�����û�ж���ͼ����ر����
//    {
//		for(int i=0;i<num;i++)
//		{
//          capture>>image02[i];
//          if(image02[i].empty())
//          {
//            capture.release();
//          }
//		}
//    }        
//    waitKey(5);
//    Mat image1[num],image2[num];   
//    Mat temp,image02temp;
//    vector<KeyPoint> keyPoint1[num], keyPoint2[num];//����ͼ�м�⵽��������
//    SurfDescriptorExtractor SurfDescriptor;//SURF���������� 
//    Mat imageDesc1[num],imageDesc2[num];
//    int bAdaptFast=0;                      //�Ƿ��������Ӧ����ֵ,����fast��ֵ��ʱ����Ҫ��̫��֡
//    vector<DMatch> matchePoints;           //ƥ���������
//	vector<vector<Point>>contours;         //����
//	while(1)
//    {
//        target.clear(); //���ο��ڴ���ͷ�
//        vector<Rect>(target).swap(target);//����ط���ʲô��˼
//		//����һ֡�����ݱ�������
//		//��һ֡ͼ
//		for(int j=0;j<num;j++)
//		{
//          image1[j]=image2[j].clone();
//          image01[j]=image02[j].clone();
//          imageDesc1[j]=imageDesc2[j].clone();
//          keyPoint1[j]=keyPoint2[j];
//		  capture>>image02[j];
//          //�Ҷ�ͼת��,�ڴ�֮ǰimage2����Ϣ�Ѿ����ݸ���image1
//          cvtColor(image02[j],image2[j],CV_BGR2GRAY);  
//          //��ȡ������   
//          FAST(image2[j], keyPoint2[j], fastThreshold);//fast��������
//          //����ƥ�����������������FAST��ֵ�Ķ�̬����
//          //��ⲻ��������ƥ��ᱨ������0�Ļ���û��Ҫ�ڽ���ƥ�䣬��Ȼ�Ǻ����ƥ��ᱨ��
//          if (keyPoint2[j].size() == 0)
//          {
//            fastThreshold -= 2;  
//            bAdaptFast = 0;//û�е�����ֵ
//            imshow("pic", image02[j]);
//            waitKey(10);
//            continue;//�����ⲻ�������㣬�Ͳ���ִ�к����������ƥ�����
//          }
//          //������������Ϊ�±ߵ�������ƥ����׼��      
//          SurfDescriptor.compute(image2[j], keyPoint2[j], imageDesc2[j]);
//          //���ƥ�������㣬����ȡ�������     
//          FlannBasedMatcher matcher;  
//          //����ƥ��
//          matcher.match(imageDesc1[j], imageDesc2[j], matchePoints, Mat());  
//          sort(matchePoints.begin(), matchePoints.end());//����������   
//         if(matchePoints.size()<100)//���������������һ�����ƣ������⵽��������С��100��
//         {
//            fastThreshold-=1;
//            if(fastThreshold <10)
//            {
//                fastThreshold =10;
//            }
//            bAdaptFast = 0;//û�е�����ֵ
//            imshow("pic", image02[j]);
//            waitKey(10);
//            continue;//��ֵû�е�����
//        }
//        if(matchePoints.size()>300)//���������������һ������
//        {
//            fastThreshold += int(matchePoints.size()/100);//�����ʽ��������ô������������
//            bAdaptFast = 0;//û�е�����ֵ
//            imshow("pic",image02[j]);
//            waitKey(10);
//            continue;
//        }
//        bAdaptFast = 1;//�����ִ�е�����˵���Ѿ�����������ֵ
//        //ɸ����ƥ��������
//		vector<Point2f> imagePoints1,imagePoints2;  
//        for(int i=0; i < matchePoints.size() * 0.5; i++)//ֻѡȡ����ƥ�䵽���������ǰ�ٷ�֮��ʮ 
//        {         
//            imagePoints1.push_back(keyPoint1[j][matchePoints[i].queryIdx].pt);       
//            imagePoints2.push_back(keyPoint2[j][matchePoints[i].trainIdx].pt);       
//        }  
//        //��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
//        Mat homo = findHomography(imagePoints1,imagePoints2,CV_RANSAC);//CV_RANSAC³���Էǳ��� 
//        //ͼ����׼  
//        Mat imageTransform1,imgpeizhun,imgerzhi;
//		//����һ֡ͼƬͨ��ͶӰӳ�����ת������һ֡������ϵ�¡�������ת���߶Ȳ����Ρ�
//        warpPerspective(image01[j], imageTransform1, homo, Size(image02[j].cols, image02[j].rows));    
//        //imshow("����͸�Ӿ���任��",imageTransform1); 
//		//ͼ����,�õ����ڵ�ǰ֡����ϵ���������ͼ��Ĳ��
//        absdiff(image02[j], imageTransform1, imgpeizhun);//ע������ͼ��ߴ�һ��Ҫһ��
//        //imshow("��׼diff",imgpeizhun);  
//        Mat imgOtsu =imgpeizhun(Rect(imgpeizhun.cols *(1-bili), imgpeizhun.rows*(1-bili),imgpeizhun.cols * (2 * bili - 1), imgpeizhun.rows*(2 * bili - 1)));
//        double MaxValue;
//        minMaxLoc(imgOtsu, 0, &MaxValue);//�Ҳ��ͼ������ֵ
//		//��������������ô�趨�ģ���������������MaxValue*0.7������Զ�̬����ô��������
//        threshold(imgpeizhun, imgerzhi, MaxValue*0.7/*- 10*/, 255.0 , CV_THRESH_BINARY);//10Ϊ�������
//        //threshold(imgpeizhun, imgerzhi, dlg->GetDlgItemInt(IDC_EDIT_DIFF), 255.0 , CV_THRESH_BINARY);
//        //imshow("��׼��ֵ��", imgerzhi);
//        image02temp = image02[j].clone();
//        cvtColor(imgerzhi, temp, CV_RGB2GRAY);//ת���ɻҶ� 
//        //������ͨ��
//        //Mat se=getStructuringElement(MORPH_RECT, Size(3,3));
//        //morphologyEx(temp, temp,MORPH_OPEN,se);
//        int dialate_size = 9;//����size,�����ֵ�Ĵ�С��Ӱ��Ч���ɣ�����˵����3,5,7��8,9
//        Mat se2=getStructuringElement(MORPH_RECT,Size(dialate_size,dialate_size));
//        morphologyEx(temp,temp,MORPH_DILATE,se2);
//        findContours(temp,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);//������е��ⲿ����
//        //����������ɸѡ
//        if (contours.size() < 1)
//        {
//            imshow("pic", image02[j]);
//            waitKey(10);
//            continue;
//        }
//        sort(contours.begin(), contours.end(), biggerSort);//������С������
//        //Ŀ��ȷ��
//        if(true)//��ѡģʽ
//        {
//            int count=(contours.size() > 12)? 12:contours.size();//Ŀ����������ಶ׽12��Ŀ��ô����
//            for (int k = 0; k < count; k++)
//            {
//                Rect bomen = boundingRect(contours[k]);//��������Ӿ���
//                //ʡ��������׼�����ı�Ե��Ч��Ϣ
//                if (bomen.x > image02temp.cols * (1 - bili) && bomen.y > image02temp.rows * (1 - bili) 
//                    && bomen.x + bomen.width < image02temp.cols * bili && bomen.y + bomen.height < image02temp.rows * bili
//                    /*&& contourArea(contours[k]) > contourArea(contours[0])/10*/
//                    && contourArea(contours[k])>20 && contourArea(contours[k])<1000)//�����������ƿ��Խ��ж�̬����
//                {         
//                    //ɸѡĿ����״
//                    if (bomen.width > bomen.height)
//                    {
//                        if (bomen.width > 2 * bomen.height)
//                        {
//                            continue;
//                        }
//                    }
//                    if (bomen.width < bomen.height)
//                    {
//                        if (2 * bomen.width < bomen.height)
//                        {
//                            continue;
//                        }
//                    }
//					
//                    Rect temp;
//                    int offset = 5;//44*44
//                    temp.x = bomen.x + bomen.width / 2 - offset;
//                    temp.y = bomen.y + bomen.height / 2 - offset;
//                    temp.width = offset * 2;
//                    temp.height = offset * 2;
//                    bomen = temp;
//
//                    rectangle(image02temp, bomen, Scalar(0,0,255), 2, 8, 0);
//                    target.push_back(bomen);
//                }
//            }
//        }
//         imshow("pic", image02temp);
//         waitKey(10);  
//		}
//    }
//}
//
//
//
//
//

