import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.MultiBoostAB;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A tuner that does cross-validated parameter selection on a
 * SingleClassiferEnhancer which uses J48 and picks the best confidence factor
 * for J48.
 */
public class J48EnhancedParameterTuner extends SingleClassifierEnhancer {
	private static final long serialVersionUID = 320723416556929258L;
	private static final String PARAMETER = "C";
	private static double[] PARAMETERS = new double[] { .05, 0.1, .15, 0.2, .25, 0.3, .35, 0.4,
			.45, 0.5 };
	private ParameterSelectedResult bestResult;

	public J48EnhancedParameterTuner() {
		setClassifier(new AdaBoostM1());
	}

	public J48EnhancedParameterTuner(SingleClassifierEnhancer baseLearner) {
		setClassifier(baseLearner);
	}

	@Override
	protected String defaultClassifierString() {
		return "weka.classifiers.meta.AdaBoostM1";
	}

	static class ParameterSelectionWorker implements Callable<ParameterSelectedResult> {
		SingleClassifierEnhancer classifier;
		Instances trainingData;
		double param;

		public ParameterSelectionWorker(SingleClassifierEnhancer classifier,
				Instances trainingData, double param) {
			this.classifier = classifier;
			this.trainingData = trainingData;
			this.param = param;
		}

		@Override
		public ParameterSelectedResult call() throws Exception {
			J48 weakLearner = new J48();
			weakLearner.setConfidenceFactor((float) param);
			classifier.setClassifier(weakLearner);

			classifier.buildClassifier(trainingData);

			Evaluation eval = new Evaluation(trainingData);
			eval.crossValidateModel(classifier, trainingData, 10, new Random(0));

			double errorRate = eval.errorRate();
			return new ParameterSelectedResult(new EvaluationResult(errorRate, null), classifier,
					PARAMETER, param);
		}

	}

	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		ExecutorService executor = Executors.newFixedThreadPool(PARAMETERS.length);
		List<Future<ParameterSelectedResult>> resultFutures = new ArrayList<Future<ParameterSelectedResult>>();

		for (double param : PARAMETERS) {
			SingleClassifierEnhancer classifier = (SingleClassifierEnhancer) Classifier
					.makeCopy(getClassifier());
			if (classifier instanceof AdaBoostM1) {
				((AdaBoostM1) classifier).setNumIterations(25);
			} else if (classifier instanceof Bagging) {
				((Bagging) classifier).setNumIterations(15);
			} else if (classifier instanceof Dagging) {
				((Dagging) classifier).setNumFolds(15);
			} else if (classifier instanceof MultiBoostAB) {
				((MultiBoostAB) classifier).setNumIterations(25);
			}

			Callable<ParameterSelectedResult> worker = new ParameterSelectionWorker(classifier,
					trainingData, param);
			resultFutures.add(executor.submit(worker));
		}

		List<ParameterSelectedResult> resultList = new ArrayList<ParameterSelectedResult>();
		for (Future<ParameterSelectedResult> futureResult : resultFutures) {
			resultList.add(futureResult.get());
		}
		executor.shutdown();
		bestResult = Collections.min(resultList);
		setClassifier(bestResult.model);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return getClassifier().distributionForInstance(instance);
	}

	public String getParam() {
		return PARAMETER;
	}

	public double getBestParamValue() {
		return bestResult.getParamValue();
	}

}
