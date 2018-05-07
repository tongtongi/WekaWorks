import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ConsistencySubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.EER;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class DenemeWeka {
    public static String result = "";
    public static String resultCfs = "";
    public static String resultConsistency = "";

    public static void main(String[] args) {
        System.out.println("HELLO");
        /*if (args.length > 0){
            System.out.println(args[0]);
            File file = new File(args[0]);
            try {
                Scanner scan = new Scanner(file);
                while (scan.hasNextLine()){
                    String s = scan.nextLine();
                    System.out.println(s);
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        System.exit(0);*/
        String type = args[1];
        int[] arr = {0,1,14, 15, 29, 42, 43, 56, 69, 70, 83, 97, 98, 104};
        HashMap<String, DataSource> dataMap = new HashMap<String, DataSource>();
        File folder = new File(args[0]);
        File[] listOfFiles = folder.listFiles();
        for (int i = 0; i < listOfFiles.length; i++) {
            DataSource ds = null;
            try {
                ds = new DataSource(listOfFiles[i].getAbsolutePath());
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (ds != null)
                dataMap.put(listOfFiles[i].getName().substring(0, 6), ds);
        }
        try {


            int count = 1;
            for (Map.Entry<String, DataSource> entry : dataMap.entrySet()) {
                Instances data = entry.getValue().getDataSet();
                Remove removeFilter = new Remove();
                removeFilter.setAttributeIndicesArray(arr);
                removeFilter.setInputFormat(data);
                data = Filter.useFilter(data, removeFilter);

                System.out.println(entry.getKey());
                System.out.println("Count : " + count);
                if(type.equalsIgnoreCase("cfs")){
                    resultCfs += entry.getKey() +"\n" + "--------RANDOM FOREST APPLIED WITH CFS SUBSET EVAL----------"+"\n";
                    Evaluation eval = applyRandomForestWithCfsSubSet(data);
                    System.out.println("Random Forest  WITH CFS  is finished");
                    resultCfs += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";




                    resultCfs += "--------J48 APPLIED WITH CFS SUBSET EVAL----------"+"\n";
                    eval = applyJ48WithCfsSubSet(data);
                    System.out.println("J48  WITH CFS  is finished");
                    resultCfs += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";

                    /*resultCfs += "--------LibSVM APPLIED WITH CFS SUBSET EVAL----------"+"\n";
                    eval = applyLibSVMWithCfsSubSet(data);
                    System.out.println("LibSVM  WITH CFS  is finished");
                    resultCfs += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";*/


                    resultCfs += "--------NAIVE BAYES APPLIED WITH CFS SUBSET EVAL----------"+"\n";
                    eval = applyNaiveBayesWithCfsSubSet(data);
                    System.out.println("Naive  WITH CFS  Bayes is finished");
                    resultCfs += eval.toSummaryString("\nResults\n======\n", false)+"\n\n"+ eval.toClassDetailsString() +"\n\n\n\n\n";

                }



                //////////////////////////////////////////////////////////////////////////////////////////////////////////////
                if(type.equalsIgnoreCase("cons")){
                    resultConsistency += entry.getKey() +"\n" + "--------RANDOM FOREST APPLIED WITH CONSISTENCY SUBSET EVAL----------"+"\n";
                    Evaluation eval = applyRandomForestWithConsistencySubSet(data);
                    System.out.println("Random Forest  WITH CONSISTENCY  is finished");
                    resultConsistency += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";

                    resultConsistency += "--------J48 APPLIED WITH CONSISTENCY SUBSET EVAL----------"+"\n";
                    eval = applyJ48WithConsistencySubSet(data);
                    System.out.println("J48  WITH CONSISTENCY  is finished");
                    resultConsistency += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";

                    /*resultConsistency += "--------LibSVM APPLIED WITH CONSISTENCY SUBSET EVAL----------"+"\n";
                    eval = applyLibSVMWithConsistencySubSet(data);
                    System.out.println("LibSVM  WITH CONSISTENCY  is finished");
                    resultConsistency += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";*/



                    resultConsistency += "--------NAIVE BAYES APPLIED WITH CONSISTENCY SUBSET EVAL----------"+"\n";
                    eval = applyNaiveBayesWithConsistencySubSet(data);
                    System.out.println("Naive Bayes  WITH CONSISTENCY  is finished");
                    resultConsistency += eval.toSummaryString("\nResults\n======\n", false)+"\n\n"+ eval.toClassDetailsString() + "\n\n\n\n\n";


                }

                //////////////////// Classification without attribute selection
                if(type.equalsIgnoreCase("none")){
                    result += entry.getKey() + "\n" + "--------RANDOM FOREST APPLIED----------" + "\n";
                    Evaluation eval = applyRandomForest(data);
                    System.out.println("Random Forest is finished");
                    result += eval.toSummaryString("\nResults\n======\n", false) + "\n\n" + eval.toClassDetailsString() + "\n";


                    result += "--------J48 APPLIED----------" + "\n";
                    eval = applyJ48(data);
                    System.out.println("J48 is finished");
                    result += eval.toSummaryString("\nResults\n======\n", false) + "\n" + "\n\n" + eval.toClassDetailsString() + "\n";


                    /*eval = applyLibSVM(data);
                    System.out.println("LibSVM is finished");
                    result += eval.toSummaryString("\nResults\n======\n", false)+"\n"+"\n\n"+ eval.toClassDetailsString() + "\n";*/


                    result += "--------NAIVE BAYES APPLIED----------"+"\n";
                    eval = applyNaiveBayes(data);
                    System.out.println("Naive Bayes is finished");
                    result += eval.toSummaryString("\nResults\n======\n", false)+"\n\n"+ eval.toClassDetailsString() +"\n\n\n\n\n";


                }


                count++;

            }


            PrintWriter out = new PrintWriter(args[0]+"/result.txt");

            PrintWriter outCfs = new PrintWriter(args[0]+"/resultCfs.txt");
            PrintWriter outConsistency = new PrintWriter(args[0]+"/resultConsistency.txt");
            try {
                if(type.equalsIgnoreCase("none"))
                    out.println(result);
                if(type.equalsIgnoreCase("cfs"))
                    outCfs.println(resultCfs);
                if(type.equalsIgnoreCase("cons"))
                    outConsistency.println(resultConsistency);
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                out.close();
                outCfs.close();
                outConsistency.close();

            }


        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static Evaluation applyJ48(Instances data) throws Exception {


        if (data.classIndex() == -1)
            data.setClassIndex(24);

        String[] options = new String[4];
        options[0] = "-C";
        options[1] = "0.25";
        options[2] = "-M";
        options[3] = "2";
        J48 tree = new J48();
        tree.setOptions(options);
        tree.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(tree, data, 10, new Random(1));

        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);

        result += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";

        return eval;
    }


    public static Evaluation applyJ48WithCfsSubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        CfsSubsetEval cfs = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);

        String[] options = new String[4];
        options[0] = "-C";
        options[1] = "0.25";
        options[2] = "-M";
        options[3] = "2";
        J48 tree = new J48();
        tree.setOptions(options);
        classifier.setClassifier(tree);
        classifier.setEvaluator(cfs);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultCfs += "\n\n" + classifier.toString();
        resultCfs += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";

        return eval;
    }

    public static Evaluation applyJ48WithConsistencySubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        ConsistencySubsetEval consistency = new ConsistencySubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);

        String[] options = new String[4];
        options[0] = "-C";
        options[1] = "0.25";
        options[2] = "-M";
        options[3] = "2";
        J48 tree = new J48();
        tree.setOptions(options);
        classifier.setClassifier(tree);
        classifier.setEvaluator(consistency);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultConsistency += "\n\n" + classifier.toString();
        resultConsistency += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";

        return eval;
    }

    public static Evaluation applyRandomForest(Instances data) throws Exception {
        if (data.classIndex() == -1)
            data.setClassIndex(24);


        RandomForest tree = new RandomForest();

        tree.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(tree, data, 10, new Random(1));

        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);

        result += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";

        return eval;
    }

    public static Evaluation applyRandomForestWithCfsSubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        CfsSubsetEval cfs = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        RandomForest tree = new RandomForest();

        classifier.setClassifier(tree);
        classifier.setEvaluator(cfs);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

//        eval.evaluateModel(classifier, data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        //System.out.println(eval.toSummaryString("\nResults\n======\n", true) + "\n");
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultCfs += "\n\n" + classifier.toString();
        resultCfs += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;

    }

    public static Evaluation applyRandomForestWithConsistencySubSet(Instances data) throws Exception {


        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        ConsistencySubsetEval consistency = new ConsistencySubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);

        RandomForest tree = new RandomForest();

        classifier.setClassifier(tree);
        classifier.setEvaluator(consistency);
        classifier.setSearch(search);
        classifier.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultConsistency += "\n\n" + classifier.toString();
        resultConsistency += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyMlp(Instances data) throws Exception {


        if (data.classIndex() == -1)
            data.setClassIndex(24);


        MultilayerPerceptron mlp = new MultilayerPerceptron();

        mlp.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.evaluateModel(mlp, data);
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);

        result += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyLibSVM(Instances data) throws Exception {


        if (data.classIndex() == -1)
            data.setClassIndex(24);


        LibSVM libSVM = new LibSVM();

        libSVM.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(libSVM, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);

        result += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyLibSVMWithCfsSubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        CfsSubsetEval cfs = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        LibSVM libSVM = new LibSVM();

        classifier.setClassifier(libSVM);
        classifier.setEvaluator(cfs);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultCfs += "\n\n" + classifier.toString();
        resultCfs += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyLibSVMWithConsistencySubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        ConsistencySubsetEval consistency = new ConsistencySubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        LibSVM libSVM = new LibSVM();

        classifier.setClassifier(libSVM);
        classifier.setEvaluator(consistency);
        classifier.setSearch(search);
        classifier.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultConsistency += "\n\n" + classifier.toString();
        resultConsistency += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applySMO(Instances data) throws Exception {


        if (data.classIndex() == -1)
            data.setClassIndex(24);


        SMO smo = new SMO();

        smo.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.evaluateModel(smo, data);
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);

        result += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applySMOWithCfsSubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        CfsSubsetEval cfs = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        SMO smo = new SMO();

        classifier.setClassifier(smo);
        classifier.setEvaluator(cfs);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultCfs += "\n\n" + classifier.toString();
        resultCfs += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applySMOWithConsistencySubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        ConsistencySubsetEval consistency = new ConsistencySubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        SMO smo = new SMO();

        classifier.setClassifier(smo);
        classifier.setEvaluator(consistency);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultConsistency += "\n\n" + classifier.toString();
        resultConsistency += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyNaiveBayes(Instances data) throws Exception {


        if (data.classIndex() == -1)
            data.setClassIndex(24);


        NaiveBayes naiveBayes = new NaiveBayes();

        naiveBayes.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(naiveBayes, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);

        result += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyNaiveBayesWithCfsSubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        CfsSubsetEval cfs = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        NaiveBayes naiveBayes = new NaiveBayes();

        classifier.setClassifier(naiveBayes);
        classifier.setEvaluator(cfs);
        classifier.setSearch(search);
        classifier.buildClassifier(data);
        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultCfs += "\n\n" + classifier.toString();
        resultCfs += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }

    public static Evaluation applyNaiveBayesWithConsistencySubSet(Instances data) throws Exception {

        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        ConsistencySubsetEval consistency = new ConsistencySubsetEval();
        BestFirst search = new BestFirst();

        if (data.classIndex() == -1)
            data.setClassIndex(24);


        NaiveBayes naiveBayes = new NaiveBayes();

        classifier.setClassifier(naiveBayes);
        classifier.setEvaluator(consistency);
        classifier.setSearch(search);
        classifier.buildClassifier(data);

        Evaluation eval = new Evaluation(data);

        eval.crossValidateModel(classifier, data, 10, new Random(1));
        ThresholdCurve curve = new ThresholdCurve();
        Instances rocPoints = curve.getCurve(eval.predictions());
        EER eer = new EER(rocPoints);
        resultConsistency += "\n\n" + classifier.toString();
        resultConsistency += "\n\nEER= " + eer.calculateEER() * 100 + "%\n";
        return eval;
    }
}
