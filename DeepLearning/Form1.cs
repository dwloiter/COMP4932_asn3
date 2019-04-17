using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace DeepLearning
{
    public partial class Form1 : Form
    {
        private const string training_inputs = "training_inputs.txt";
        private const string training_results = "training_results.txt";
        private const string validation_inputs = "validation_inputs.txt";
        private const string validation_results = "validation_results.txt";
        private const string test_inputs = "test_inputs.txt";
        private const string test_results = "test_results.txt";

        private const int image_size = 28 * 28;
        private const int number_of_intermediates = 30;
        private const int number_of_numbers = 10;

        private List<double[]> trainingInputs = new List<double[]>();
        private List<int> trainingResults = new List<int>();
        private List<double[]> validationInputs = new List<double[]>();
        private List<int> validationResults = new List<int>();
        private List<double[]> testInputs = new List<double[]>();
        private List<int> testResults = new List<int>();

        private int numberOfLayers;
        private int[] sizes;
        private List<double[]> biases;
        private List<double[,]> weights;

        Random random;

        private delegate void SafeCallDelegate(string text);

        private void WriteTextSafe(string text)
        {
            if (Result.InvokeRequired)
            {
                var d = new SafeCallDelegate(WriteTextSafe);
                Invoke(d, new object[] { text });
            }
            else
            {
                Result.Text += text;
            }
        }

        public Form1()
        {
            InitializeComponent();
            LoadData();

            random = new Random();

            sizes = new int[3];
            sizes[0] = image_size;
            sizes[1] = number_of_intermediates;
            sizes[2] = number_of_numbers;

            biases = new List<double[]>();
            weights = new List<double[,]>();

            Init(sizes);
        }

        private void LoadData()
        {
            FileStream stream = File.Open(training_inputs, FileMode.Open);
            StreamReader streamReader = new StreamReader(stream);
            while (true)
            {
                string line = streamReader.ReadLine();
                if (line == null || line.Length == 0)
                {
                    break;
                }
                string[] image = line.Split(' ');
                trainingInputs.Add(new double[image_size]);
                for (int i = 0; i < image_size; ++i)
                {
                    trainingInputs[trainingInputs.Count - 1][i] = double.Parse(image[i]);
                }
            }
            streamReader.Close();
            stream.Close();

            stream = File.Open(training_results, FileMode.Open);
            streamReader = new StreamReader(stream);
            while (true)
            {
                string line = streamReader.ReadLine();
                if (line == null || line.Length == 0)
                {
                    break;
                }
                trainingResults.Add(int.Parse(line));
            }
            streamReader.Close();
            stream.Close();

            stream = File.Open(validation_inputs, FileMode.Open);
            streamReader = new StreamReader(stream);
            while (true)
            {
                string line = streamReader.ReadLine();
                if (line == null || line.Length == 0)
                {
                    break;
                }
                string[] image = line.Split(' ');
                validationInputs.Add(new double[image_size]);
                for (int i = 0; i < image_size; ++i)
                {
                    validationInputs[validationInputs.Count - 1][i] = double.Parse(image[i]);
                }
            }
            streamReader.Close();
            stream.Close();

            stream = File.Open(validation_results, FileMode.Open);
            streamReader = new StreamReader(stream);
            while (true)
            {
                string line = streamReader.ReadLine();
                if (line == null || line.Length == 0)
                {
                    break;
                }
                validationResults.Add(int.Parse(line));
            }
            streamReader.Close();
            stream.Close();

            stream = File.Open(test_inputs, FileMode.Open);
            streamReader = new StreamReader(stream);
            while (true)
            {
                string line = streamReader.ReadLine();
                if (line == null || line.Length == 0)
                {
                    break;
                }
                string[] image = line.Split(' ');
                testInputs.Add(new double[image_size]);
                for (int i = 0; i < image_size; ++i)
                {
                    testInputs[testInputs.Count - 1][i] = double.Parse(image[i]);
                }
            }
            streamReader.Close();
            stream.Close();

            stream = File.Open(test_results, FileMode.Open);
            streamReader = new StreamReader(stream);
            while (true)
            {
                string line = streamReader.ReadLine();
                if (line == null || line.Length == 0)
                {
                    break;
                }
                testResults.Add(int.Parse(line));
            }
            streamReader.Close();
            stream.Close();
        }

        private void Init(int[] sizes)
        {
            numberOfLayers = sizes.Length;
            for (int i = 1; i < numberOfLayers; ++i)
            {
                biases.Add(new double[sizes[i]]);

                for (int j = 0; j < sizes[i]; ++j)
                {
                    biases[biases.Count - 1][j] = random.NextDouble() / 10;
                }
            }

            for (int i = 0; i < numberOfLayers - 1; ++i)
            {
                weights.Add(new double[sizes[i + 1], sizes[i]]);

                for (int row = 0; row < sizes[i + 1]; ++row)
                {
                    for (int col = 0; col < sizes[i]; ++col)
                    {
                        weights[weights.Count - 1][row, col] = random.NextDouble() / 10;
                    }
                }
            }
        }

        private void SGD(int epochs, int miniBatchSize, int eta)
        {
            int nTest = testInputs.Count;
            int nTraining = trainingInputs.Count;

            for (int j = 0; j < epochs; ++j)
            {
                shuffleTraining();
                for (int i = 0; i < nTraining; i += miniBatchSize)
                {
                    // update mini batch
                    update_mini_batch(i, i + miniBatchSize, eta);
                }

                WriteTextSafe("\nEpoch " + j + ": " + evaluate() + " / " + testInputs.Count);
            }
        }

        private void shuffleTraining()
        {
            for (int i = 0; i < trainingInputs.Count; ++i)
            {
                int index = random.Next(trainingInputs.Count);
                if (i != index)
                {
                    double[] temp = trainingInputs[i];
                    trainingInputs[i] = trainingInputs[index];
                    trainingInputs[index] = temp;

                    int r = trainingResults[i];
                    trainingResults[i] = trainingResults[index];
                    trainingResults[index] = r;
                }
            }
        }

        private void update_mini_batch(int startIndex, int endIndex, float eta)
        {
            List<double[]> nabla_b = new List<double[]>();
            List<double[,]> nabla_w = new List<double[,]>();
            int mini_batch_count = endIndex - startIndex;
            for (int b = 0; b < biases.Count(); ++b)
            {
                nabla_b.Add(new double[biases[b].Length]);
            }
            for (int w = 0; w < weights.Count(); ++w)
            {
                nabla_w.Add(new double[weights[w].GetLength(0), weights[w].GetLength(1)]);
            }

            List<double[]> delta_nabla_b;
            List<double[,]> delta_nabla_w;

            for (int x = startIndex; x < endIndex; ++x)
            {
                backprop(trainingInputs[x], trainingResults[x], out delta_nabla_b, out delta_nabla_w);

                for (int i = 0; i < nabla_b.Count; ++i)
                {
                    for (int j = 0; j < nabla_b[i].Length; ++j)
                    {
                        nabla_b[i][j] += delta_nabla_b[i][j];
                    }
                }
                
                for (int k = 0; k < nabla_w.Count; ++k)
                {
                    for (int i = 0; i < nabla_w[k].GetLength(0); ++i)
                    {
                        for (int j = 0; j < nabla_w[k].GetLength(1); ++j)
                        {
                            nabla_w[k][i, j] += delta_nabla_w[k][i, j];
                        }
                    }
                }
            }

            for (int x = 0; x < weights.Count; ++x)
            {
                for (int y = 0; y < weights[x].GetLength(0); ++y)
                {
                    for (int z = 0; z < weights[x].GetLength(1); ++z)
                    {
                        weights[x][y, z] -= (eta / mini_batch_count) * nabla_w[x][y, z];
                    }
                }
            }
            
            for (int i = 0; i < biases.Count; ++i)
            {
                for (int j = 0; j < biases[i].Length; ++j)
                {
                    biases[i][j] -= (eta / mini_batch_count) * nabla_b[i][j];
                }
            }
        }

        private void backprop(double[] x, int y, out List<double[]> delta_nabla_b, out List<double[,]> delta_nabla_w)
        {
            delta_nabla_b = new List<double[]>();
            delta_nabla_w = new List<double[,]>();

            for (int b = 0; b < biases.Count(); ++b)
            {
                delta_nabla_b.Add(new double[biases[b].Length]);
            }
            for (int w = 0; w < weights.Count(); ++w)
            {
                delta_nabla_w.Add(new double[weights[w].GetLength(0), weights[w].GetLength(1)]);
            }

            double[] activation = x;
            List<double[]> activations = new List<double[]>();
            activations.Add(activation);
            List<double[]> zs = new List<double[]>();

            for (int i = 0; i < weights.Count; ++i)
            {
                double[] z = new double[weights[i].GetLength(0)];
                double[] temp = new double[weights[i].GetLength(0)];
                for (int j = 0; j < weights[i].GetLength(0); ++j)
                {
                    z[j] = dot(activation, weights[i], j) + biases[i][j];
                    temp[j] = sigmoid(z[j]);
                }
                zs.Add(z);
                activation = temp;
                activations.Add(temp);
            }

            double[] delta = cost_derivative(activations[activations.Count - 1], y);
            for (int i = 0; i < delta.Length; ++i)
            {
                delta[i] *= sigmoid_prime(zs[zs.Count - 1][i]);
            }
            delta_nabla_b[delta_nabla_b.Count - 1] = delta;
            delta_nabla_w[delta_nabla_w.Count - 1] = mul(delta, activations[activations.Count - 2]);

            for (int i = 2; i < numberOfLayers; ++i)
            {
                double[] z = zs[zs.Count - i];
                double[] sp = new double[z.Length];
                for (int j = 0; j < sp.Length; ++j)
                {
                    sp[j] = sigmoid_prime(z[j]);
                }
                delta = mul(weights[weights.Count - i + 1], delta, sp);
                delta_nabla_b[delta_nabla_b.Count - i] = delta;
                delta_nabla_w[delta_nabla_w.Count - i] = mul(delta, activations[activations.Count - i - 1]);
            }
        }

        double[] dot(double[] l, double[,] r)
        {
            double[] ret = new double[r.GetLength(0)];
            for (int i = 0; i < r.GetLength(0); ++i)
            {
                for (int j = 0; j < r.GetLength(1); ++j)
                {
                    ret[i] += l[j] * r[i, j];
                }
            }
            return ret;
        }

        double dot(double[] l, double[,] r, int rIndex)
        {
            double ret = 0;
            for (int i = 0; i < l.Length; ++i)
            {
                ret += l[i] * r[rIndex, i];
            }
            return ret;
        }

        double dot(double[] l, double[] r)
        {
            double ret = 0;
            for (int i = 0; i < l.Length; ++i)
            {
                ret += l[i] * r[i];
            }

            return ret;
        }
        
        double[,] mul(double[] vec, double[] matTrans)
        {
            double[,] ret = new double[vec.Length, matTrans.Length];

            for (int row = 0; row < ret.GetLength(0); ++row)
            {
                for (int col = 0; col < ret.GetLength(1); ++col)
                {
                    ret[row, col] = vec[row] * matTrans[col];
                }
            }

            return ret;
        }
        
        double[] mul(double[,] matTrans, double[] vec, double[] sp)
        {
            double[] ret = new double[matTrans.GetLength(1)];
            for (int i = 0; i < matTrans.GetLength(1); ++i)
            {
                for (int j = 0; j < matTrans.GetLength(0); ++j)
                {
                    ret[i] += vec[j] * matTrans[j, i] * sp[i];
                }
            }
            return ret;
        }

        double sigmoid(double z)
        {
            double ret = 1.0 / (1.0 + Math.Exp(-z));
            return ret;
        }

        double sigmoid_prime(double z)
        {
            return sigmoid(z) * (1-sigmoid(z));
        }
        
        double[] cost_derivative(double[] outputActivations, int y)
        {
            double[] ret = new double[outputActivations.Length];

            for (int i = 0; i < outputActivations.Length; ++i)
            {
                ret[i] = outputActivations[i] - (y == i ? 1.0 : 0.0);
            }

            return ret;
        }

        double[] feedforward(double[] a)
        {
            double[] tmp = a;
            for(int w = 0; w < weights.Count; ++w)
            {
                tmp = dot(tmp, weights[w]);
                
                for(int i = 0; i < tmp.Length; ++i)
                {
                    tmp[i] = sigmoid(tmp[i] + biases[w][i]);
                }
            }
            
            return tmp;
        }
        //double[,] dot(double[,] a, double[,] b)
        //{
        //        double[,] result = new double[a.GetLength(0), b.GetLength(1)];

        //        for (int i = 0; i < a.GetLength(0); ++i)
        //        {
        //            for (int j = 0; j < b.GetLength(1); ++j)
        //            {
        //                result [i, j] = 0;
        //                for (int k = 0; k < a.GetLength(1); ++k)
        //               {
        //                    result[i, j] += a[i, k] * b[k, j];
        //                }
        //            }
        //        }
        //        return result;
        //}
        int evaluate()
        {
            int sum = 0;
            for(int x = 0 ; x < testInputs.Count; ++x)
            {
                double[] result = feedforward(testInputs[x]);
                int maxIndex = 0;
                double max = result[0];
                for (int i = 1; i < result.Length; ++i)
                {
                    if (result[i] > max)
                    {
                        max = result[i];
                        maxIndex = i;
                    }
                }
                if (testResults[x] == maxIndex)
                {
                    sum++;
                }
            }

            return sum;
        }

        Thread t;
        private void button1_Click(object sender, EventArgs e)
        {
            t = new Thread(() => SGD(30, 10, 3));
            t.Start();
        }

        protected override void OnFormClosing(System.Windows.Forms.FormClosingEventArgs e)
        {
            t.Abort();
            base.OnFormClosing(e);
        }
    }
}
