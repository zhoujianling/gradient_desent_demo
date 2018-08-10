package com.company;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import static java.lang.Math.E;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static java.lang.System.exit;


public class Solver {

    private List<Point> rawData = new ArrayList<>();
    private List<Point> trainData = new ArrayList<>();
    private List<Point> testData = new ArrayList<>();
    private Model model = null;
    private Function<Model, Double> loss = null;

    public Solver(String csvPath) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(csvPath));
        while (scanner.hasNextLine()) {
            String[] xy = scanner.nextLine().split(",");
            rawData.add(new Point(Double.valueOf(xy[0]), Double.valueOf(xy[1])));
        }
    }

    private Function<Model, Double> mse = (m) -> {
        double lossSum = 0.0;
        for (Point p : trainData) {
            double diff = m.val(p.x) - p.y;
            lossSum += (diff * diff);
        }
        return lossSum / 2.0;
    };

    private void divide(float ratio4Train) {
        trainData.clear();
        testData.clear();
        if (ratio4Train <= 0) throw new IllegalArgumentException("Ratio <= 0");
        int testCount = (int) (rawData.size() * (1 - ratio4Train));
        Random rand = new Random(System.currentTimeMillis());
        Set<Integer> exclusiveIndices4Test = new HashSet<>();
        while (exclusiveIndices4Test.size() < testCount) {
            int index = rand.nextInt(rawData.size());
            if (! exclusiveIndices4Test.contains(index)) {
                testData.add(rawData.get(index));
                exclusiveIndices4Test.add(index);
            }
        }
        for (int i = 0; i < rawData.size(); i ++) {
            if (! exclusiveIndices4Test.contains(i)) {
                trainData.add(rawData.get(i));
            }
        }
    }

    private void train() {
        System.out.println("Train data size: " + trainData.size());
        System.out.println("Test data size: " + testData.size());
//        model = new PolyModel(4);
        model = new GaussianModel(5);
        loss = mse;
        // ==========================================================
        for (int i = 0; i < 10000; i ++) {
            double lossVal = loss.apply(model);
            double[] gradVal = model.grad(trainData);
            System.out.println(String.format("Iter: %d, loss: %f ", i, lossVal));
            System.out.println(String.format("Theta: %f, %f, %f", model.theta[0], model.theta[1], model.theta[2]));
            System.out.println(String.format("Grad: %f, %f, %f\n", gradVal[0], gradVal[1], gradVal[2]));
            if (Double.isNaN(lossVal)) {
                model.randomize(); i = 0;
                continue;
            }
            for (int j = 0; j < gradVal.length; j ++) {
                double delta = model.rate(j) * gradVal[j];
                model.theta[j] -= delta;
            }
//            if (lossVal < 1.06) break;
        }
        System.out.println(String.format("Theta: %f, %f, %f", model.theta[0], model.theta[1], model.theta[2]));
    }

    private void validate() {
        double RMSE = 0.0;
        for (Point p : testData) {
            double diff = model.val(p.x) - p.y;
            RMSE += (diff * diff);
        }
        RMSE /= testData.size();
        RMSE = sqrt(RMSE);
        System.out.println("RMSE: " + RMSE);
    }

    private void plot() {
        XYChart chart = QuickChart.getChart(
                "Result", "X", "Y", "y(x)",
                trainData.stream().map(point -> point.x).collect(Collectors.toList()),
                trainData.stream().map(point -> point.y).collect(Collectors.toList()));

        double[] xPoints = new double[150];
        double[] yPoints = new double[150];
        for (int i = 0; i < 150; i ++) {
            xPoints[i] = i * 10.0 / 150;
            yPoints[i] = model.val(xPoints[i]);
        }
        chart.addSeries("model", xPoints, yPoints);

        new SwingWrapper<XYChart>(chart).displayChart();
    }

    public void solve() {
        divide(0.8f);
        train();
        validate();
        plot();
    }

    public static void main(String[] args) throws FileNotFoundException {
	// write your code here
        if (args.length < 1) {
            System.out.println("Usage: java -jar GradientDesent.jar data.csv");
            exit(0);
        }
        new Solver(args[0]).solve();
    }

    private static class Point {
        double x;
        double y;
        public Point(double x, double y) {this.x = x; this.y = y;}

    }

    private static abstract class Model {
        double theta[] = null;
        abstract double val(double x);
        abstract double[] grad(List<Point> trainData);
        abstract void randomize();
        abstract double rate(int i);
    }

    private static class PolyModel extends Model{

        public PolyModel(int n) {
            if (n < 2) throw new IllegalArgumentException("n MUST be larger than 2.");
            theta = new double[n];
            randomize();
        }

        double val(double x) {
            double result = 0.0;
            for (int i = 0; i < theta.length; i ++) {
                result += theta[i] * pow(x, i);
            }
            return result;
        }

        @Override
        double[] grad(List<Point> trainData) {
            double []gradVec = new double[theta.length];
            for (int i = 0; i < gradVec.length; i ++) {
                gradVec[i] = 0.0;
                Random r = new Random();
                List<Point> data = new ArrayList<>();
                for (int k = 0; k < 50; k ++)
                    data.add(trainData.get(r.nextInt(trainData.size())));
                for (Point p : data) {
                    double diff = val(p.x) - p.y;
                    gradVec[i] += (diff * pow(p.x, i));
                }
            }
            return gradVec;
        }

        @Override
        void randomize() {
            Random rand = new Random(System.currentTimeMillis());
            for (int i = 0; i < theta.length; i ++) {
                theta[i] = rand.nextDouble() ;
            }
        }

        @Override
        double rate(int i) {
            return 0.00000002;
        }
    }

    private static class GaussianModel extends Model{

        /**
         * f(x) = a * e ^ (- (x - μ)^2 / σ^2)
         * (a, μ, σ2) <<----
         * @param n number of gaussian function
         */
        public GaussianModel(int n) {
            if (n < 1) throw new IllegalArgumentException("n MUST be larger than 1.");
            theta = new double[n * 3];
            randomize();
        }

        @Override
        double val(double x) {
            double result = 0.0;
            for (int i = 0; i < theta.length / 3; i ++) {
                double alpha = theta[i * 3 + 0];
                double miu = theta[i * 3 + 1];
                double sigma2 = theta[i * 3 + 2];
                result += (alpha * pow(E, - pow((x - miu), 2) / sigma2 / 2));
            }
            return result;
        }

        @Override
        double[] grad(List<Point> trainData) {
            double[] gradVec = new double[theta.length];
            for (int i = 0; i < theta.length / 3; i ++) {
                gradVec[i * 3 + 0] = 0;
                gradVec[i * 3 + 1] = 0;
                gradVec[i * 3 + 2] = 0;
                double alpha = theta[i * 3 + 0];
                double miu = theta[i * 3 + 1];
                double sigma2 = theta[i * 3 + 2];
                Random r = new Random();
                List<Point> stochasticData = new ArrayList<>();
                for (int k = 0; k < 30; k ++)
                    stochasticData.add(trainData.get(r.nextInt(trainData.size())));
                for (Point p : stochasticData) {
                    double val = val(p.x);
                    gradVec[i * 3 + 0] += 2
                            * (val - p.y)
                            * (pow(E, - pow((p.x - miu), 2) / sigma2 / 2));
                    gradVec[i * 3 + 1] += (2
                            * alpha
                            * (val - p.y)
                            * pow(E, - pow((p.x - miu), 2) / sigma2 / 2)
                            * ((p.x - miu) / sigma2));
                    gradVec[i * 3 + 2] += (2
                            * alpha
                            * (val - p.y)
                            * pow(E, - pow((p.x - miu), 2) / sigma2 / 2)
                            * (pow((p.x - miu), 2) / pow(sigma2, 2) / 2)); //把sigma平方当成了一个整体
                }
            }
            return gradVec;
        }

        @Override
        void randomize() {
            Random rand = new Random(System.currentTimeMillis());
            for (int i = 0; i < theta.length / 3; i ++) {
                theta[i * 3 + 0] = rand.nextDouble();
                theta[i * 3 + 1] = rand.nextDouble() * 5;
                theta[i * 3 + 2] = rand.nextDouble();
            }
        }

        @Override
        double rate(int i) {
            if (i % 3 == 0) {
                return 0.0005;
            } else if (i % 3 == 1) { // miu
                return 0.0005;
            } else {
                return 0.00005;
            }
        }

        public String toString() {
            StringBuilder builder = new StringBuilder("Theta: ");
            for (double t : theta) {
                builder.append(t);
                builder.append(", ");
            }
            builder.append("\nGrad: ");
            return builder.toString();
        }
    }
}
