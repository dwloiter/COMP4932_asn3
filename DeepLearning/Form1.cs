using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
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


        private List<double[]> nabla_b;
        private List<double[,]> nabla_w;
        private List<double[,]> mini_batch;

        Random random;

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
            nabla_b = new List<double[]>();
            nabla_w = new List<double[,]>();
            mini_batch = new List<double[,]>();

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
            for (int i = 1; i < numberOfLayers - 1; ++i)
            {
                biases.Add(new double[sizes[i]]);

                for (int j = 0; j < sizes[i]; ++j)
                {
                    biases[biases.Count - 1][j] = random.NextDouble();
                }
            }

            for (int i = 0; i < numberOfLayers - 1; ++i)
            {
                weights.Add(new double[sizes[i + 1], sizes[i]]);

                for (int row = 0; row < sizes[i + 1]; ++row)
                {
                    for (int col = 0; col < sizes[i]; ++col)
                    {
                        weights[weights.Count - 1][row, col] = random.NextDouble();
                    }
                }
            }
        }

       
        private void update_mini_batch(int startIndex, int endIndex, float eta)
        {
           for(int b = 0; b < biases.Count(); ++b)
           {
                nabla_b.Add(new double[biases[b].Length]);
           }
           for(int w = 0; w < weights.Count(); ++w)
           {
                nabla_w.Add(new double[weights[w].GetLength(0), weights[w].GetLength(1)]);
           }

           
           List<double[]> delta_nabla_b;
           List<double[,]> delta_nabla_w;

           for(int x = 0; x < mini_batch.Count; ++x)
           {
                for(int y = 0; y < mini_batch.Count-1; ++y)
                {
                    delta_nabla_b = backprop(x, y);
                    delta_nabla_w = backprop(x, y);
                    
                    
                    nabla_b[x][y] += delta_nabla_b[x][y];
                    nabla_w[x, y][x, y] += delta_nabla_w[x, y][x, y];


                }
           }
           
           for(int x = 0; x < weights.Count; ++x)
           {
                 for(int y = 0; y < weights.Count-1; ++y)
                 {
                    weights[x, y] -= (eta/mini_batch.Count) * nabla_w[x,y];
                 }
           }

           for(int i = 0 ; i < biases.Count; ++i)
           {
                biases[i] -= (eta/mini_batch.Count) * nabla_b[i];
           }

        }

        private void backprop(int x, int y, out List<double[]> nabla_b, out List<double[,]> nabla_w)
        {

        }

        
        double sigmoid(int z)
        {
            return 1.0/ (1.0 + Math.Pow(Math.E, -z));
        }

        double sigmoid_prime(int z)
        {
            return sigmoid(z) * (1-sigmoid(z));
        }
        
        double[] cost_derivative(double[] output_activations, double[] y)
        {
            double[] result = new double[output_activations.Length];
            for(int i = 0; i < output_activations.Length; ++i)
            {
                result[i] = output_activations[i] - y[i];
            }
            return result;
        }

        void feedforward(out double[] a)
        {
            for(int b = 0; b < biases.Count; ++b)
            {
                for(int w = 0; w < weights.Count; ++w)
                {
                    double[] tmp = dot(weights[w], a);
                
                    for(int i = 0; i < tmp.Length; ++i)
                    {
                        a = sigmoid(tmp[i] + biases[b]);

                    }
                    
                }                       
            }
        }
        double[,] dot(double[,] a, double[,] b)
        {
                double[,] result = new double[a.GetLength(0), b.GetLength(1)];

                for (int i = 0; i < a.GetLength(0); ++i)
                {
                    for (int j = 0; j < b.GetLength(1); ++j)
                    {
                        result [i, j] = 0;
                        for (int k = 0; k < a.GetLength(1); ++k)
                        {
                            result[i, j] += a[i, k] * b[k, j];
                        }
                    }
                }
                return result;
        }
        double evaluate(List<double[,]> test_data)
        {   
            
            for(int x = 0 ; x < test_data.Count; ++x)
            {
                double[] result = feedforward(test_data[i]);
                
            }

        }    
    }
}
