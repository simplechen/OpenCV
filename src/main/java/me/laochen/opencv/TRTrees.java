package me.laochen.opencv;

import org.bytedeco.javacpp.opencv_ml.ANN_MLP;
import org.bytedeco.javacpp.opencv_ml.DTrees;
import org.bytedeco.javacpp.opencv_ml.KNearest;
import org.bytedeco.javacpp.opencv_ml.NormalBayesClassifier;
import org.bytedeco.javacpp.opencv_ml.RTrees;
import org.bytedeco.javacpp.opencv_ml.SVM;
import org.bytedeco.javacpp.opencv_ml.SVMSGD;

/**
 * 随机树  http://www.cnblogs.com/starfire86/articles/5219597.html
 * http://blog.csdn.net/bleakie/article/details/54576201
 * http://blog.csdn.net/bleakie/article/details/54576201
 * @author simple
 *
 */
public class TRTrees {
	public static void main(String[] args) {
		RTrees rt= RTrees.create();//  随机森林
		DTrees.create();//决策树		
		NormalBayesClassifier.create();//贝叶斯分类器		
		KNearest.create();//K最近邻
		
		ANN_MLP.create();//人工神经网络
		
		SVM.create();//支持向量机
		
		SVMSGD.create();//随机梯度下降SGD
		
	}

}
