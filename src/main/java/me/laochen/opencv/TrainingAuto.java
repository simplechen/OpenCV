package me.laochen.opencv;

import static org.bytedeco.javacpp.opencv_core.CV_32FC1;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_ml.ROW_SAMPLE;

import java.io.File;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ml.SVM;


/**
 * 车牌号码分类 训练
 * 根据C++ 版本   http://blog.csdn.net/ap1005834/article/details/51313602 进行调整
 * 
 * 3.2 C++ 版本  http://answers.opencv.org/question/65764/svm-training-data-error/
 * 
 * 官方文档  https://docs.opencv.org/3.0-beta/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
 * 
 * 矩阵操作: http://blog.csdn.net/iracer/article/details/51296631
 * @author laochen
 *
 */
public class TrainingAuto {
	protected static final String PATH_POSITIVE = "./static/svm_auto/posdata/train";
    protected static final String PATH_NEGATIVE = "./static/svm_auto/negdata/train";
    protected static final String XML = "./static/svm_auto/test.xml";
    protected static final String FILE_TEST_POS = "./static/svm_auto/posdata/test/000-(0).jpg";
    protected static final String FILE_TEST_NEG = "./static/svm_auto/negdata/test/000-(0).jpg";
    
	
	public static void main(String[] args) {
		
		Mat trainingImages = new Mat();//用于训练的图片 含 好的样本和坏的样本  
        
        //加载正样本http://blog.csdn.net/z914022466/article/details/52709981
        int posLength=0;
        for ( File file : new File( PATH_POSITIVE ).listFiles() ) {
            Mat img = imread( file.getAbsolutePath());
            trainingImages.push_back(getMat( img));
            posLength++;
        }
        
        //加载负样本
        int negLength = 0;
        for ( File file : new File( PATH_NEGATIVE ).listFiles() ) {
            Mat img = imread( file.getAbsolutePath() );
            trainingImages.push_back( getMat( img) );
            negLength++;
        }
        
        System.err.println(posLength+"--"+negLength); 
        
        /**
         http://blog.csdn.net/Eroslol/article/details/52525541
         Mat M(7,7,CV_32FC2,Scalar(1,3));
			解释如下：创建一个M矩阵，7行7列，类型为CV_32F，C2表示有2个通道。
			Scalar(1,3)是对矩阵进行初始化赋值。第一个通道全为1，第2个通道全为3。
         */
        //构建一维矩阵  与训练数据相应的标签  
        Mat trainingLabels = Mat.zeros(posLength+negLength,1,CV_32SC1).asMat(); //与训练数据相应的标签    CV_32SC1       CV_32FC1  
        for(int i=0;i<trainingLabels.arrayHeight();i++) {       	
        	//这个地方有问题
    		if(i<posLength) {
    			trainingLabels.ptr(i, 0).fill(1);
    		} else {
    			trainingLabels.ptr(i, 0).fill(2);
    		}
        }
    	
    	System.err.println(trainingImages);
    	System.err.println(trainingLabels);
    	
    	
    	SVM clasificador = SVM.create();    	
    	clasificador.setKernel(SVM.LINEAR);
    	clasificador.setGamma(3.0f);
    	clasificador.setC(0.1);
		clasificador.train(trainingImages, ROW_SAMPLE,trainingLabels); //ROW_SAMPLE  COL_SAMPLE The type of responses cannot be float or double.		
		File sFile= new File(XML);
		if(sFile.exists()) {
			sFile.delete();
		}
		clasificador.save(XML);//将训练结果存盘  
 
	    Mat dst = imread(FILE_TEST_POS);
	    dst = getMat(dst);
        float flag = clasificador.predict( dst );
        System.err.println(flag);
	}
	
	//灰度处理
	 protected static Mat getMat( Mat img ) {
       Mat timg = new Mat();
       /**
        * RGB模式就是，色彩数据模式，R在高位，G在中间，B在低位。BGR正好相反
        * 例如，如果色彩数据是24位，对于RGB模式，就是高8位是R，中间8位是G，低8位是B。一个色彩数据共24位，3个字节。
        */
       opencv_imgproc.cvtColor(img, timg, opencv_imgproc.COLOR_RGB2GRAY);//  COLOR_BGR2GRAY
       /**
         	在无需复制数据的前提下改变2D矩阵的形状和通道数或其中之一。
			C++: Mat Mat::reshape(int cn, int rows=0) const
			参数：
			cn – 新的通道数。若cn=0，那么通道数就保持不变。
			rows –新的行数。 若rows = 0, 那么行数保持不变。
			该方法为*this元素创建新的矩阵头。这新的矩阵头尺寸和通道数或其中之一发生改变，在以下的情况任意组合都是有可能的：
			ü  新的矩阵没有新增或减少元素。通常，rows*cols*channels()在转换过程中保持一致。.
			ü  无数据的复制。也就是说，这是一个复杂度为 O(1)的操作。通常，如果该操作改变行数或透过其他方式改变元素行索引，那么矩阵必定是连续的。参见Mat::isContinuous()。
			例如，有一存储了STL向量的三维点集，你想用3xN的矩阵来完成下面的操作：
			std::vector<Point3f> vec;
			...
			Mat pointMat = Mat(vec). //把向量转化成Mat, 复杂度为O(1)的运算
			reshape(1). // 从Nx1的3通道矩阵得出Nx3 的单通道矩阵
			//同样是复杂度为O(1)的运算
			t(); // 最后转置Nx3 的矩阵
			//这个过程要复制所有的元素
			 
			而Mat是与图像宽高对应的矩阵，因此在输入前我们需要使用reshape(1,1)方法把矩阵拉伸成向量
        */
       timg = timg.reshape( 1, 1 );
       
       //http://blog.csdn.net/qq_22764813/article/details/52135686
       //http://blog.csdn.net/sunshine_in_moon/article/details/45174597
       /**
        m  目标矩阵。如果m的大小与原矩阵不一样，或者数据类型与参数不匹配，那么在函数convertTo内部会先给m重新分配空间。
		 rtype 指定从原矩阵进行转换后的数据类型，即目标矩阵m的数据类型。当然，矩阵m的通道数应该与原矩阵一样的。如果rtype是负数，
		 	那么m矩阵的数据类型应该与原矩阵一样。
        */
       timg.convertTo( timg,  CV_32FC1);//CV_32FC1将CV_32FC1的指针类型定义为float，将CV_64FC1的指针类型定义为double
       return timg;
   }

}
