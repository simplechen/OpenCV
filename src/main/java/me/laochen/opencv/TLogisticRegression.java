package me.laochen.opencv;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_32S;
//import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_ml.ROW_SAMPLE;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.FileStorage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_ml.LogisticRegression;

/**
 * 逻辑回归 :  http://www.cnblogs.com/denny402/p/5032490.html
 * 矩阵操作: http://blog.csdn.net/iracer/article/details/51296631
 * @author simple
 *
 */
public class TLogisticRegression {

	public static void main(String[] args) {

		FileStorage fs = new FileStorage("./static/logistic/data01.xml", FileStorage.READ);
		Mat data = fs.get("datamat").mat();
		Mat labels = fs.get("labelsmat").mat();
		fs.release();

		// 转换成float型
		data.convertTo(data, CV_32F);
		labels.convertTo(labels, CV_32F);
		
		System.err.println("读取了 " + data.rows() + " 行数据.");

		//用于训练模型的数据
		Mat data_train = new Mat();
		Mat labels_train = new Mat();
		
		//用于测试模型的数据
		Mat data_test = new Mat();
		Mat labels_test = new Mat();

		for (int i = 0; i < data.rows(); i++) {
			if (i % 2 == 0) {
				data_train.push_back(data.row(i));
				labels_train.push_back(labels.row(i));
			} else {
				data_test.push_back(data.row(i));
				labels_test.push_back(labels.row(i));
			}
		}

		System.err.println("训练数据:" + data_train.rows());
		System.err.println("测试数据:" + data_test.rows());

		// 显示样本图片
		showImage(data_train, 28, "train_data.jpg");
		showImage(data_test, 28, "test_data.jpg");

		// 创建分类器并设置参数
		LogisticRegression lr1 = LogisticRegression.create();
		lr1.setLearningRate(0.001);
		lr1.setIterations(10);
		lr1.setRegularization(LogisticRegression.REG_L2);
		lr1.setTrainMethod(LogisticRegression.BATCH);
		lr1.setMiniBatchSize(1);

		// 训练分类器
		lr1.train(data_train, ROW_SAMPLE, labels_train);
		lr1.save("./static/logistic/r.xml");		
		
		// 预测
		Mat responses = new Mat();
		lr1.predict(data_test, responses, 0);

		//labels_test 为已知的标签
		labels_test.convertTo(labels_test, CV_32S); //转换为整型
		dump(labels_test);
		
		
		System.err.println("--------------------");
		
		//responses 为通过逻辑回归预测出来的标签
		dump(responses);
		
		System.err.println(calculateAccuracyPercent(labels_test, responses));
	}

	protected static float calculateAccuracyPercent(Mat original, Mat predicted) {
		//return 100 * (float) opencv_core.countNonZero(opencv_core.subtract(original, predicted).asMat()) / predicted.rows();		
//		opencv_core.InputArray
//		opencv_ml.InputArray ia = new InputArray();
//		return 100 * (float) opencv_core.countNonZero(opencv_core.addPut(original, predicted)) / predicted.rows();

		return 100 * (float) opencv_core.countNonZero(predicted) / opencv_core.countNonZero(original);
	}

	// 将向量转化成图片矩阵并显示
	private static void showImage(Mat data, int columns, String name) {
		Mat bigImage = new Mat();
		for (int i = 0; i < data.rows(); ++i) {
			bigImage.push_back(data.row(i).reshape(0, columns));
		}
		// 保存图片
		imwrite("./static/logistic/" + name, bigImage);
	}

	public static int getMatElement(Mat img, int row, int col, int channel) {
		// 获取字节指针
		BytePointer bytePointer = img.ptr(row, col);
		int value = bytePointer.get(channel);
		if (value < 0) {
			value = value + 256;
		}
		return value;
	}
	
	public static void dump(Mat mat) {
		System.err.print("[");
		for(int i=0;i<mat.rows();i++) {
			for(int j=0;j<mat.arrayWidth();j++) {
				BytePointer el = mat.ptr(i, j);
				System.err.print(el.get(0)+",");
			}
		}
		System.err.print("],");
	}
}
