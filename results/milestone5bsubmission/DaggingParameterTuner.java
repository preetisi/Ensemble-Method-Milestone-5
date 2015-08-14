import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A tuner that does cross-validated parameter selection on AdaBoostM1 and J48
 * and picks the best confidence factor for J48.
 */
public class DaggingParameterTuner extends SingleClassifierEnhancer {
	private static final long serialVersionUID = 320723416556929258L;
	private static final String PARAMETER = "C";
	private ParameterSelectedResult bestResult;

	public DaggingParameterTuner() {
		setClassifier(new AdaBoostM1());
	}

	@Override
	protected String defaultClassifierString() {

		return "weka.classifiers.meta.AdaBoostM1";
	}

	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		List<ParameterSelectedResult> resultList = new ArrayList<ParameterSelectedResult>();

		for (double param : new Double[] { 0.1, 0.2, 0.3, 0.4, 0.5 }) {
			AdaBoostM1 classifier = new AdaBoostM1();
			classifier.setNumIterations(25);

			J48 weakLearner = new J48();
			weakLearner.setConfidenceFactor((float) param);
			classifier.setClassifier(weakLearner);

			classifier.buildClassifier(trainingData);

			Evaluation eval = new Evaluation(trainingData);
			eval.crossValidateModel(classifier, trainingData, 10, new Random(0));
			// eval.evaluateModel(classifier, testingData);

			double errorRate = eval.errorRate();
			resultList.add(new ParameterSelectedResult(new EvaluationResult(errorRate, null),
					classifier, PARAMETER, param));
		}
		bestResult = Collections.min(resultList);
		setClassifier(bestResult.model);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return bestResult.model.distributionForInstance(instance);
	}

	public String getParam() {
		return PARAMETER;
	}

	public double getBestParamValue() {
		return bestResult.getParamValue();
	}

}
