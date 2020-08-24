

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <iostream>

using namespace cv;
using namespace std;


//对轮廓按面积降序排列  
bool biggerSort(vector<Point> v1, vector<Point> v2)  
{  
    return contourArea(v1)>contourArea(v2);  
}  
void main()
{
    VideoCapture capture;//视频源
    Mat image01,image02;
    vector<Rect> target;//画出矩形框
    int fastThreshold = 100;//这个值是通过大量是测试得来的吗？？
    float bili = 0.8;//由于两幅图配准，边缘不会一致，因此对原图大小0.9的比例中搜索检测到的目标

    //capture.open(0);//打开摄像头
    capture.open(0);//打开视频
    if (capture.isOpened())//如果相机打开，但是没有读出图像，则关闭相机
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

    vector<KeyPoint> keyPoint1, keyPoint2;//两幅图中检测到的特征点

    SurfDescriptorExtractor SurfDescriptor;//SURF特征点描述 
    Mat imageDesc1,imageDesc2;
    int bAdaptFast=0;                      //是否完成自适应调阈值,调节fast阈值的时候不需要隔太多帧
    vector<DMatch> matchePoints;           //匹配的特征点
	vector<vector<Point>>contours;        //轮廓
	while (1)
    {
        target.clear(); //矩形框内存的释放
        vector<Rect>(target).swap(target);//这个地方是什么意思
		//将上一帧的数据保存下来
        image1=image2.clone();
        image01=image02.clone();
        imageDesc1=imageDesc2.clone();
        keyPoint1=keyPoint2;
        //后一帧图
        int num = 5;  
        if (!bAdaptFast)//没有完成自适应阈值调节
        {
            num = 1;
        }
        bool flag = true;
        while (num-- && flag)//完成了自适应调节就循环5次，没有完成自适应调剂就循环1次
        {
            capture >> image02;//
            //imageSave=image02.clone();//备份到save图中
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
        if (!flag)//如果读取到的图片为空就直接跳出循环
        {
            break;
        }           
        //灰度图转换,在此之前image2的信息已经传递给了image1
        cvtColor(image02,image2,CV_BGR2GRAY);  
        //提取特征点   
        FAST(image2, keyPoint2, fastThreshold);//fast特征点检测
        //根据匹配的特征点数量进行FAST阈值的动态调整
        //检测不到特征点匹配会报错，等于0的话就没必要在进行匹配，不然那后面的匹配会报错
        if (keyPoint2.size() == 0)
        {
            fastThreshold -= 2;  
            bAdaptFast = 0;//没有调好阈值
            imshow("pic", image02);
            waitKey(10);
            continue;//如果检测不到特征点，就不会执行后面的特征点匹配程序
        }
        //特征点描述，为下边的特征点匹配做准备      
        SurfDescriptor.compute(image2, keyPoint2, imageDesc2);
        //获得匹配特征点，并提取最优配对     
        FlannBasedMatcher matcher;  
        //特征匹配
        matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());  
        sort(matchePoints.begin(), matchePoints.end());//特征点排序   
        
        if (matchePoints.size()<100)//对特征点的数量做一个限制，如果检测到的特征点小于100，
        {
            fastThreshold-=1;
            if(fastThreshold <10)
            {
                fastThreshold =10;
            }
            bAdaptFast = 0;//没有调好阈值
            imshow("pic", image02);
            waitKey(10);
            continue;//阈值没有调整好
        }
        if(matchePoints.size()>300)//对特征点的数量做一个限制
        {
            fastThreshold += int(matchePoints.size()/100);//这个公式是随便给的么？？？？？？
            bAdaptFast = 0;//没有调好阈值
            imshow("pic",image02);
            waitKey(10);
            continue;
        }
        bAdaptFast = 1;//如果能执行到这里说明已经调整好了阈值
        //筛除误匹配特征点
		vector<Point2f> imagePoints1,imagePoints2;  
        for(int i=0; i < matchePoints.size()* 0.5; i++)//只选取所有匹配到的特征点的前百分之五十 
        {         
            imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);       
            imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);       
        }  
        //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
        Mat homo = findHomography(imagePoints1,imagePoints2,CV_RANSAC);//CV_RANSAC鲁棒性非常好 
        //图像配准  
        Mat imageTransform1,imgpeizhun,imgerzhi;
		//将上一帧图片通过投影映射矩阵转换到这一帧的坐标系下。具有旋转，尺度不变形。
        warpPerspective(image01, imageTransform1,homo,Size(image02.cols, image02.rows));    
        //imshow("经过透视矩阵变换后",imageTransform1); 
		//图像差分,得到基于当前帧坐标系下面的两幅图像的差别
        absdiff(image02, imageTransform1, imgpeizhun);//注意两幅图像尺寸一定要一致
        imshow("配准diff",imgpeizhun);  
        Mat imgOtsu =imgpeizhun(Rect(imgpeizhun.cols *(1-bili), imgpeizhun.rows*(1-bili),imgpeizhun.cols * (2 * bili - 1), imgpeizhun.rows*(2 * bili - 1)));
        double MaxValue;
        minMaxLoc(imgOtsu, 0, &MaxValue);//找差分图里最大的值
		//这个差分容限是怎么设定的？？？？？？？？MaxValue*0.7这个可以动态调整么？？？？
        threshold(imgpeizhun, imgerzhi, MaxValue*0.7/*- 10*/, 255.0 , CV_THRESH_BINARY);//10为差分容限
        //threshold(imgpeizhun, imgerzhi, dlg->GetDlgItemInt(IDC_EDIT_DIFF), 255.0 , CV_THRESH_BINARY);
        //imshow("配准二值化", imgerzhi);
        image02temp = image02.clone();
        cvtColor(imgerzhi, temp, CV_RGB2GRAY);//转换成灰度 
        //检索连通域
        //Mat se=getStructuringElement(MORPH_RECT, Size(3,3));
        //morphologyEx(temp, temp,MORPH_OPEN,se);
        int dialate_size=9;//膨胀size,这个数值的大小会影响效果吧？还是说随便给3,5,7，8,9
        Mat se2=getStructuringElement(MORPH_RECT,Size(dialate_size,dialate_size));
        morphologyEx(temp,temp,MORPH_DILATE,se2);
        findContours(temp,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);//检测所有的外部轮廓
        //轮廓数量的筛选
        if (contours.size() < 1)
        {
            imshow("pic", image02);
            waitKey(10);
            continue;
        }
        sort(contours.begin(), contours.end(), biggerSort);//轮廓大小的排序
        //目标确定
        if(true)//点选模式
        {
            int count=(contours.size() > 12)? 12:contours.size();//目标数量，最多捕捉12个目标么？？
            for (int k = 0; k < count; k++)
            {
                Rect bomen = boundingRect(contours[k]);//轮廓的外接矩形
                //省略由于配准带来的边缘无效信息
                if (bomen.x > image02temp.cols * (1 - bili) && bomen.y > image02temp.rows * (1 - bili) 
                    && bomen.x + bomen.width < image02temp.cols * bili && bomen.y + bomen.height < image02temp.rows * bili
                    /*&& contourArea(contours[k]) > contourArea(contours[0])/10*/
                    && contourArea(contours[k])>20 && contourArea(contours[k])<1000)//这里的面积限制可以进行动态调整
                {         
                    //筛选目标形状
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


////////////////////////////////////////隔帧检测1和6,2和7,3和8，4和9,5和10;6和11,7和12,8和13,9和14,10和15///////////////////////////////////////////
//#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp> 
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//#define  num 5
//int   NUM=5;
////对轮廓按面积降序排列  
//bool biggerSort(vector<Point> v1, vector<Point> v2)  
//{  
//    return contourArea(v1)>contourArea(v2);  
//}  
//void main()
//{
//    VideoCapture capture;//视频源
//    Mat image01[num],image02[num];
//    vector<Rect> target;//画出矩形框
//    int fastThreshold = 100;//这个值是通过大量是测试得来的吗
//    float bili = 0.9;//由于两幅图配准，边缘不会一致，因此对原图大小0.9的比例中搜索检测到的目标
//    //capture.open(0);//打开摄像头
//    capture.open("move.mp4");//打开视频
//    if (capture.isOpened())//如果相机打开，但是没有读出图像，则关闭相机
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
//    vector<KeyPoint> keyPoint1[num], keyPoint2[num];//两幅图中检测到的特征点
//    SurfDescriptorExtractor SurfDescriptor;//SURF特征点描述 
//    Mat imageDesc1[num],imageDesc2[num];
//    int bAdaptFast=0;                      //是否完成自适应调阈值,调节fast阈值的时候不需要隔太多帧
//    vector<DMatch> matchePoints;           //匹配的特征点
//	vector<vector<Point>>contours;         //轮廓
//	while(1)
//    {
//        target.clear(); //矩形框内存的释放
//        vector<Rect>(target).swap(target);//这个地方是什么意思
//		//将上一帧的数据保存下来
//		//后一帧图
//		for(int j=0;j<num;j++)
//		{
//          image1[j]=image2[j].clone();
//          image01[j]=image02[j].clone();
//          imageDesc1[j]=imageDesc2[j].clone();
//          keyPoint1[j]=keyPoint2[j];
//		  capture>>image02[j];
//          //灰度图转换,在此之前image2的信息已经传递给了image1
//          cvtColor(image02[j],image2[j],CV_BGR2GRAY);  
//          //提取特征点   
//          FAST(image2[j], keyPoint2[j], fastThreshold);//fast特征点检测
//          //根据匹配的特征点数量进行FAST阈值的动态调整
//          //检测不到特征点匹配会报错，等于0的话就没必要在进行匹配，不然那后面的匹配会报错
//          if (keyPoint2[j].size() == 0)
//          {
//            fastThreshold -= 2;  
//            bAdaptFast = 0;//没有调好阈值
//            imshow("pic", image02[j]);
//            waitKey(10);
//            continue;//如果检测不到特征点，就不会执行后面的特征点匹配程序
//          }
//          //特征点描述，为下边的特征点匹配做准备      
//          SurfDescriptor.compute(image2[j], keyPoint2[j], imageDesc2[j]);
//          //获得匹配特征点，并提取最优配对     
//          FlannBasedMatcher matcher;  
//          //特征匹配
//          matcher.match(imageDesc1[j], imageDesc2[j], matchePoints, Mat());  
//          sort(matchePoints.begin(), matchePoints.end());//特征点排序   
//         if(matchePoints.size()<100)//对特征点的数量做一个限制，如果检测到的特征点小于100，
//         {
//            fastThreshold-=1;
//            if(fastThreshold <10)
//            {
//                fastThreshold =10;
//            }
//            bAdaptFast = 0;//没有调好阈值
//            imshow("pic", image02[j]);
//            waitKey(10);
//            continue;//阈值没有调整好
//        }
//        if(matchePoints.size()>300)//对特征点的数量做一个限制
//        {
//            fastThreshold += int(matchePoints.size()/100);//这个公式是随便给的么？？？？？？
//            bAdaptFast = 0;//没有调好阈值
//            imshow("pic",image02[j]);
//            waitKey(10);
//            continue;
//        }
//        bAdaptFast = 1;//如果能执行到这里说明已经调整好了阈值
//        //筛除误匹配特征点
//		vector<Point2f> imagePoints1,imagePoints2;  
//        for(int i=0; i < matchePoints.size() * 0.5; i++)//只选取所有匹配到的特征点的前百分之五十 
//        {         
//            imagePoints1.push_back(keyPoint1[j][matchePoints[i].queryIdx].pt);       
//            imagePoints2.push_back(keyPoint2[j][matchePoints[i].trainIdx].pt);       
//        }  
//        //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
//        Mat homo = findHomography(imagePoints1,imagePoints2,CV_RANSAC);//CV_RANSAC鲁棒性非常好 
//        //图像配准  
//        Mat imageTransform1,imgpeizhun,imgerzhi;
//		//将上一帧图片通过投影映射矩阵转换到这一帧的坐标系下。具有旋转，尺度不变形。
//        warpPerspective(image01[j], imageTransform1, homo, Size(image02[j].cols, image02[j].rows));    
//        //imshow("经过透视矩阵变换后",imageTransform1); 
//		//图像差分,得到基于当前帧坐标系下面的两幅图像的差别
//        absdiff(image02[j], imageTransform1, imgpeizhun);//注意两幅图像尺寸一定要一致
//        //imshow("配准diff",imgpeizhun);  
//        Mat imgOtsu =imgpeizhun(Rect(imgpeizhun.cols *(1-bili), imgpeizhun.rows*(1-bili),imgpeizhun.cols * (2 * bili - 1), imgpeizhun.rows*(2 * bili - 1)));
//        double MaxValue;
//        minMaxLoc(imgOtsu, 0, &MaxValue);//找差分图里最大的值
//		//这个差分容限是怎么设定的？？？？？？？？MaxValue*0.7这个可以动态调整么？？？？
//        threshold(imgpeizhun, imgerzhi, MaxValue*0.7/*- 10*/, 255.0 , CV_THRESH_BINARY);//10为差分容限
//        //threshold(imgpeizhun, imgerzhi, dlg->GetDlgItemInt(IDC_EDIT_DIFF), 255.0 , CV_THRESH_BINARY);
//        //imshow("配准二值化", imgerzhi);
//        image02temp = image02[j].clone();
//        cvtColor(imgerzhi, temp, CV_RGB2GRAY);//转换成灰度 
//        //检索连通域
//        //Mat se=getStructuringElement(MORPH_RECT, Size(3,3));
//        //morphologyEx(temp, temp,MORPH_OPEN,se);
//        int dialate_size = 9;//膨胀size,这个数值的大小会影响效果吧？还是说随便给3,5,7，8,9
//        Mat se2=getStructuringElement(MORPH_RECT,Size(dialate_size,dialate_size));
//        morphologyEx(temp,temp,MORPH_DILATE,se2);
//        findContours(temp,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);//检测所有的外部轮廓
//        //轮廓数量的筛选
//        if (contours.size() < 1)
//        {
//            imshow("pic", image02[j]);
//            waitKey(10);
//            continue;
//        }
//        sort(contours.begin(), contours.end(), biggerSort);//轮廓大小的排序
//        //目标确定
//        if(true)//点选模式
//        {
//            int count=(contours.size() > 12)? 12:contours.size();//目标数量，最多捕捉12个目标么？？
//            for (int k = 0; k < count; k++)
//            {
//                Rect bomen = boundingRect(contours[k]);//轮廓的外接矩形
//                //省略由于配准带来的边缘无效信息
//                if (bomen.x > image02temp.cols * (1 - bili) && bomen.y > image02temp.rows * (1 - bili) 
//                    && bomen.x + bomen.width < image02temp.cols * bili && bomen.y + bomen.height < image02temp.rows * bili
//                    /*&& contourArea(contours[k]) > contourArea(contours[0])/10*/
//                    && contourArea(contours[k])>20 && contourArea(contours[k])<1000)//这里的面积限制可以进行动态调整
//                {         
//                    //筛选目标形状
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

