import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.Serializable;

import weka.classifiers.Classifier;

/**
 * Encapsulates the result of building and evaluating a model.
 */
public class ClassifierResult implements Comparable<ClassifierResult>, Serializable {
	private static final long serialVersionUID = -2400991440188107235L;
	private final double errorRate;
	protected final Classifier model;
	private String classifierName;
	private ClassifierResult ratioResult;
	private final double[] predictions;

	public ClassifierResult(EvaluationResult evalResult, Classifier model) {
		this.errorRate = evalResult.error;
		this.predictions = evalResult.predictions;
		this.model = model;
		this.classifierName = model.getClass().getSimpleName();
	}

	public String getName() {
		return classifierName;
	}

	public void setRatioResult(ClassifierResult nbResult) {
		this.ratioResult = nbResult;
	}

	public double getErrorRate() {
		return errorRate;
	}

	public double getErrorRatio() {
		if (ratioResult != null) {
			return errorRate / ratioResult.errorRate;
		} else {
			throw new UnsupportedOperationException("No NB result to get ratio");
		}
	}

	/**
	 * Outputs the model to the given path.
	 */
	public void outputModel(String modelOutputPath) throws FileNotFoundException, IOException {
		// serialize weka output
		// instead of 4 lines just write this line to serialize
		// source: http://weka.wikispaces.com/Serialization

		// weka.core.SerializationHelper.write("/some/where/j48.model", cls);

		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelOutputPath));
		oos.writeObject(model);
		oos.flush();
		oos.close();
	}

	@Override
	public int compareTo(ClassifierResult o) {
		if (ratioResult != null) {
			return Double.valueOf(getErrorRatio()).compareTo(o.getErrorRatio());
		} else {
			return Double.valueOf(getErrorRate()).compareTo(o.getErrorRate());
		}
	}

	public String toString() {
		return String.format("{Name: %s, ErrorRate: %f, ErrorRatio: %s}", getName(),
				getErrorRate(), ratioResult == null ? "?" : getErrorRatio());
	}

	public void outputPredictions(PrintStream out) {
		for (double prediction : predictions) {
			out.println(prediction);
		}
	}
}
