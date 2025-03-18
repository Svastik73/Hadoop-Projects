import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class MatrixMultiply {

    // Mapper class
    public static class MatrixMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Text outputKey = new Text();
        private Text outputValue = new Text();
        
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // Get filename to determine which matrix this is
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String filename = fileSplit.getPath().getName();
            
            // Parse the input line
            String line = value.toString();
            String[] parts = line.split(",");
            
            if (parts.length >= 3) {
                int row = Integer.parseInt(parts[0].trim());
                int col = Integer.parseInt(parts[1].trim());
                double val = Double.parseDouble(parts[2].trim());
                
                // If this is from matrix A
                if (filename.startsWith("A")) {
                    // For each element in matrix A, emit for all possible multiplications
                    // The key is (row of A, all columns of B)
                    // m is the number of columns in matrix B
                    int m = context.getConfiguration().getInt("m", 0);
                    
                    for (int j = 0; j < m; j++) {
                        outputKey.set(row + "," + j);
                        // Value format: "A,column,value"
                        outputValue.set("A," + col + "," + val);
                        context.write(outputKey, outputValue);
                    }
                } 
                // If this is from matrix B
                else if (filename.startsWith("B")) {
                    // For each element in matrix B, emit for all possible multiplications
                    // The key is (all rows of A, column of B)
                    // n is the number of rows in matrix A
                    int n = context.getConfiguration().getInt("n", 0);
                    
                    for (int i = 0; i < n; i++) {
                        outputKey.set(i + "," + col);
                        // Value format: "B,row,value"
                        outputValue.set("B," + row + "," + val);
                        context.write(outputKey, outputValue);
                    }
                }
            }
        }
    }
    
    // Reducer class
    public static class MatrixReducer extends Reducer<Text, Text, Text, Text> {
        private Text result = new Text();
        
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Parse the key to get row and column of the result
            String[] keyParts = key.toString().split(",");
            int resultRow = Integer.parseInt(keyParts[0]);
            int resultCol = Integer.parseInt(keyParts[1]);
            
            // Store values from matrix A and B
            Map<Integer, Double> aValues = new HashMap<>();
            Map<Integer, Double> bValues = new HashMap<>();
            
            // Collect values from both matrices
            for (Text value : values) {
                String[] valueParts = value.toString().split(",");
                String matrix = valueParts[0];
                int pos = Integer.parseInt(valueParts[1]);
                double val = Double.parseDouble(valueParts[2]);
                
                if (matrix.equals("A")) {
                    aValues.put(pos, val);
                } else if (matrix.equals("B")) {
                    bValues.put(pos, val);
                }
            }
            
            // Calculate the dot product
            double dotProduct = 0.0;
            for (Map.Entry<Integer, Double> aEntry : aValues.entrySet()) {
                int k = aEntry.getKey();
                double aVal = aEntry.getValue();
                
                if (bValues.containsKey(k)) {
                    double bVal = bValues.get(k);
                    dotProduct += aVal * bVal;
                }
            }
            
            // Only output non-zero results
            if (dotProduct != 0.0) {
                result.set(Double.toString(dotProduct));
                context.write(key, result);
            }
        }
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        
        // Set the dimensions of the matrices
        // These should be provided as command-line arguments
        // args[0]: input dir for matrix A
        // args[1]: input dir for matrix B
        // args[2]: output dir
        // args[3]: number of rows in matrix A (n)
        // args[4]: number of columns in matrix A / rows in matrix B (k)
        // args[5]: number of columns in matrix B (m)
        
        if (args.length != 6) {
            System.err.println("Usage: MatrixMultiply <input_A> <input_B> <output> <n> <k> <m>");
            System.exit(1);
        }
        
        int n = Integer.parseInt(args[3]);
        int k = Integer.parseInt(args[4]);
        int m = Integer.parseInt(args[5]);
        
        conf.setInt("n", n);
        conf.setInt("k", k);
        conf.setInt("m", m);
        
        Job job = Job.getInstance(conf, "Matrix Multiplication");
        job.setJarByClass(MatrixMultiply.class);
        job.setMapperClass(MatrixMapper.class);
        job.setReducerClass(MatrixReducer.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        // Input paths for matrix A and B
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileInputFormat.addInputPath(job, new Path(args[1]));
        
        // Output path
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}