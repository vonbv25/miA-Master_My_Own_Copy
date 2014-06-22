package mia.MySample;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.math.*;
import org.apache.mahout.vectorizer.encoders.AdaptiveWordValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
/**
 * Created by von on 6/22/14.
 */
public class SpamFilteringLR {
    //
    public static void writeDataToFile (List<Vector> docs,String fileName,
                                        FileSystem fs, Configuration conf) throws IOException
    {
        Path path = new Path(fileName);
        SequenceFile.Writer writer= new SequenceFile.Writer(fs,conf,path,
                LongWritable.class, VectorWritable.class);
        long key = 0;
        VectorWritable vec = new VectorWritable();
        for (Vector data:docs) {
            vec.set(data);
            writer.append(new LongWritable(key++),vec);
        }
        writer.close();
    }

    public static Vector encode_text(String document,String label) throws IOException{
        FeatureVectorEncoder encoder= new AdaptiveWordValueEncoder("text");
        encoder.setProbes(2);
        Analyzer analyzer= new StandardAnalyzer(Version.LUCENE_30);
        StringReader in = new StringReader(document);
        TokenStream ts= analyzer.tokenStream("content",in);
        CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
        ts.reset();
        Vector v1= new RandomAccessSparseVector(1000);
        while(ts.incrementToken()){
            char[] termBuff= termAtt.buffer();
            int termLen=termAtt.length();
            String w = new String(termBuff,0,termLen);
            encoder.addToVector(w,1,v1);
        }
        ts.end();
        ts.close();
        return new NamedVector(v1, label);

    }

    public static void main(String[] args) throws IOException{

        BufferedReader reader = new BufferedReader(new FileReader(
                SpamFilteringLR.class.getClassLoader().getResource("SMSSpamCollection").getFile()
        ));
        List<Vector> data = new ArrayList<Vector>();
        String line;
        while ((line = reader.readLine())!= null) {
            if(line.startsWith("ham")) {
                data.add(encode_text(line,"ham"));
            }
            else
            {
                data.add(encode_text(line,"spam"));
            }
        }
        File testData = new File("testdata");

        if (!testData.exists()) {
            testData.mkdir();
        }
        testData = new File("testdata/data");
        if (testData.exists()) {
            testData.mkdir();
        }
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        writeDataToFile(data,"testdata/data/vector.model",fs,conf);

        SequenceFile.Reader read =
                new SequenceFile.Reader(fs,new Path("testdata/data/vector.model"),conf);

//
        LongWritable key = new LongWritable();
        VectorWritable value = new VectorWritable();
// List<NamedVector> v = new ArrayList<NamedVector>();
        AdaptiveLogisticRegression lr = new AdaptiveLogisticRegression(

                2,1000,new L1()
        );
        while (read.next(key,value)) {
            NamedVector v = (NamedVector) value.get();
            System.out.println("Training...");
            lr.train("spam".equals(v.getName()) ? 1 : 0, v);

        }
        lr.close();
        File modelData = new File("model");

        if (!modelData.exists()) {
            modelData.mkdir();
        }
        modelData = new File("model/data");
        if (modelData.exists()) {
            modelData.mkdir();
        }

        String model_path = modelData.getPath();

        System.out.println("saving the model..");
        ModelSerializer.writeBinary(model_path,lr.getBest().getPayload().getLearner());

        InputStream in = new FileInputStream(model_path);
        CrossFoldLearner best = ModelSerializer.readBinary(in, CrossFoldLearner.class);
        in.close();

        SequenceFile.Reader read1 =
                new SequenceFile.Reader(fs,new Path("testdata/data/vector.model"),conf);

        System.out.println("Proceeding to testing of data");
        double correct = 0;
        int total = 0;
        while (read1.next(key, value)) {
            total++;
            NamedVector v = (NamedVector) value.get();
            int expected = "spam".equals(v.getName()) ? 1 : 0;

            Vector p = new DenseVector(2);
            best.classifyFull(p, v);
            int cat = p.maxValueIndex();
            if (cat == expected) {
                correct++;
            }
            System.out.println("Testing Classifiers..");
        }
        double cd = correct;
        double td = total;
        System.out.println("Classifier saved in MiA-master/model");
        System.out.println( "Accuracy: " +  ( cd / td ));
        System.out.println("AUC: + " +best.auc());
    }

}
