public class EvaluationResult {

	final double error;
	final double[] predictions;

	EvaluationResult(double error, double[] predictions) {
		this.error = error;
		this.predictions = predictions;
	}
}
