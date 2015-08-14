import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;

public class ParallelVoter extends Vote {
	private static final long serialVersionUID = 4562505417966868645L;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		Instances newData = new Instances(data);
		newData.deleteWithMissingClass();

		m_Random = new Random(getSeed());

		ExecutorService executor = Executors.newFixedThreadPool(getClassifiers().length);
		for (int i = 0; i < m_Classifiers.length; i++) {
			BuildClassifierWorker buildWorker = new BuildClassifierWorker(i, data);
			executor.execute(buildWorker);
		}
		executor.shutdown();
		while (!executor.isTerminated()) {
			Thread.sleep(100);
		}
	}

	class BuildClassifierWorker implements Runnable {
		private Classifier classifier;
		private Instances data;

		public BuildClassifierWorker(int classifierIndex, Instances data) {
			this.classifier = getClassifier(classifierIndex);
			this.data = data;
		}

		@Override
		public void run() {
			try {
				classifier.buildClassifier(data);
			} catch (Exception e) {
				System.err.println("FAILED! " + classifier);
			}
		}

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return super.distributionForInstance(instance);
	}

	@Override
	protected double[] distributionForInstanceAverage(Instance instance) throws Exception {
		return super.distributionForInstanceAverage(instance);
	}

}
