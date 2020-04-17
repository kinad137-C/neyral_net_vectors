
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
//#include <cmath>
//for checking in(cassert)

//#include<cstdlib>
#include "crtdbg.h"
using namespace std;





// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

class TrainingData
{
public:
  TrainingData(const string filename);
  bool isEof(void) { return m_trainingDataFile.eof(); }
  void getTopology(vector<unsigned>& topology);

  // Returns the number of input values read from the file:
  unsigned getNextInputs(vector<double>& inputVals);
  unsigned getTargetOutputs(vector<double>& targetOutputVals);

private:
  ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned>& topology)
{
  string line;
  string label;

  getline(m_trainingDataFile, line);
  stringstream ss(line);
  ss >> label;
  if (this->isEof() || label.compare("topology:") != 0) 
  {
    abort();
  }

  while (!ss.eof()) 
  {
    unsigned n;
    ss >> n;
    topology.push_back(n);
  }

  return;
}

TrainingData::TrainingData(const string filename)
{
  m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals)
{
  inputVals.clear();

  string line;
  getline(m_trainingDataFile, line);
  stringstream ss(line);

  string label;
  ss >> label;
  if (label.compare("in:") == 0) 
  {
    double oneValue;
    while (ss >> oneValue) 
    {
      inputVals.push_back(oneValue);
    }
  }

  return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals)
{
  targetOutputVals.clear();

  string line;
  getline(m_trainingDataFile, line);
  stringstream ss(line);

  string label;
  ss >> label;
  if (label.compare("out:") == 0) 
  {
    double oneValue;
    while (ss >> oneValue) 
    {
      targetOutputVals.push_back(oneValue);
    }
  }

  return targetOutputVals.size();
}





struct Connection
{
  double weight;
  double deltaWeight;
};

class Neuron; //just a definition yet


/*
alllows to put two subscripts, first one for the layerNumber and 
the second one for the specific number of neuron
*/
typedef vector<Neuron> Layer; 



/***************************class Neuron*******************************/
/*
class Neuron is responsible for doing the math
(division of responsibility here between classes)
*/

class Neuron
{
public:
  /*
  the minimum amount of information we need to tell the neuron about 
  the next layer in oder to it to do its job is the number of neurons in the next layer
  */
  Neuron(unsigned numOutputs, unsigned m_myIndex);
  void feedForward(const Layer& prevLayer);
  void setOutputVal(double m_outputVal) { this->m_outputVal = m_outputVal; }
  double getOutputVal(void) { return this->m_outputVal; }
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer& nextLayer);
  void updateInputWeights(Layer& prevLayer);

  ~Neuron();

private:
  /*
  static member of the class rather than dynamic member of each object
  */
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  static double eta; // [0.0...1.0] overall net training rate
  static double alpha; // [0.0.. n] multiplier of last weight change (momentum)

  /*
  rand() / double(RAND_MAX) = {0,1}*(within 0 to 1 range)
  */
  static double randomWeight(void) { return rand() / double(RAND_MAX); }

  double sumDOW(const Layer& nextLayer) const;

  double m_outputVal = 0.0;
  double m_gradient = 0.0;

  /*
  we gonna be using this vector to store the weights for all output connections that neuron has,
  and that would be it, but we will reach a point when we need also to store the changing weight
  (that is something that momentum calculations uses to implement momentum)
  */
  
  vector<Connection> m_outputWeights;
  unsigned m_myIndex;
};
double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of the last deltaWeight, [0.0 .. n]

Neuron::Neuron(unsigned numOutputs, unsigned m_myIndex)
{
  this->m_myIndex = m_myIndex;
  /*c for connections*/
  for (unsigned c = 0; c < numOutputs; c++)
  {
    m_outputWeights.push_back(Connection());
    /*
    we could put just rand()_func in here, but it would to dull
    */
    m_outputWeights.back().weight = randomWeight();
  }
}

void Neuron::feedForward(const Layer& prevLayer)//math part output=f(sum(i)+i+w+b)
{
  double sum = 0.0;
  //Sum the previous layer's outputs(witch are our inputs)
  //Include the bias node from the previous layer
  for (unsigned n = 0; n < prevLayer.size(); ++n)
  {
    sum += prevLayer[n].m_outputVal * prevLayer[n].m_outputWeights[m_myIndex].weight;
  }
  //activation or transfer function
  m_outputVal = Neuron::transferFunction(sum);

}

void Neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal - m_outputVal;
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
  double dow = sumDOW(nextLayer);
  m_gradient = dow + Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
  /*
  the weights to be updated are in the ConnectionContainer in
  the neurons in the preceding layer
  */
  for (unsigned n = 0; n < prevLayer.size(); ++n)
  {
    Neuron& neuron = prevLayer[n];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
    double newDeltaWeight =
      //individuall input, magnified by the gradient and train rate:
      /*
      n(eta) - overall net learning rate
      0.0 - slow learner
      0.2 - medium learner
      1.0 - reckless lerner
      a(alpha) - momentum
      0.0 - no momentum
      0.5 moderate momentum
      */
      eta
      * neuron.getOutputVal()
      * m_gradient
      //also add momemtum = a fraction of the previous delta weight
      + alpha
      * oldDeltaWeight;
    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }

}



Neuron::~Neuron()
{
}

double Neuron::transferFunction(double x)//tanh(could any other as well)
{
  //tanh - output range[-1.0... 1.0]
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)//the actual derivative of tanh(x) is 1 - (tanh(x))^2
{
  return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
  double sum = 0.0;
  //Sum our contributions of the errors at the nodes we feed at the next layer
  for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
  {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }
  return sum;
}





/***************************class Net**********************************/
/*
class Net is more interested in getting the loops through  the layers and neurons 
*/


class Net
{
public:
  Net(const vector<unsigned>& topology);
  void feedForward(const vector<double>& inputVals);
  void backProp(const vector<double>& targetVals);
  void getResults(vector<double>& resultVals) /* const*/;
  double getRecentAverageError(void) const { return m_recentAverageError; }
  ~Net();

private:
  vector<Layer> m_layers; //m_layers[layerNum][neyronNum]
  double m_error = 0.0;
  double m_recentAverageError = 0.0;
  double m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over;
};



Net::Net(const vector<unsigned>& topology)
{
  unsigned numLayers = topology.size(); //just for convenience
  /*
  loop is going through and on each iteration a 
  new layer is to be created (just as a creating an array but using STL)
  */
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
  {
    /*
    we are fullfilling the m_layers vector(container) by pushing 
    another vector Layer at the back of vector m_layers every iteration
    (again just like a list or an array but within STL)
    */
    m_layers.push_back(Layer());
    /*
    keep in mind that there is a different number of outputs for hidden and output layers,
    output layer don't have any
    */


    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
    /*
     as we have made a new Layer, now fill it with neurons, 
     and add a bias neuron to the layer
    */
    for (unsigned neuronNum = 0; neuronNum /* <= one more place for bias neuron*/ <= topology[layerNum]; ++neuronNum)
    {
      /*
      using .back here is ingenious touch, really, just think about it...
      we don't have to use iterators for checking the needed layer, we are just 
      pushing neurons to the most recent one 
      */
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));
      //cout << "made a neuron!!!" << endl;
    }
    // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
    m_layers.back().back().setOutputVal(1.0);
  }

}

void Net::feedForward(const vector<double>& inputVals)
{

  /*
  assert that something is to be true and during program_running if it's not... 
  you will find out(runtime error message)
  so for instance at this poin it would be wise to check if 
  the number of elements in inputVals is the same as the number of input neurons that we have
  */
  //m_layers[0] - it's also a vector(but of neurons)
  //.size()-1 is because of bias neuron
  assert(inputVals.size() == m_layers[0].size() - 1);


  // Assing (latch) the input values into the input neurons
  for (unsigned i = 0; i < inputVals.size(); ++i)
  {
    m_layers[0][i].setOutputVal(inputVals[i]);
  }
  // Forward propagate
  //starting with the 1 because inputs([0]) are already set
  for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
  {
    /*
class needs to apply func to update all input values, but to update input values
it needs to ask the neuron form previous layer what there output values are

===> it needs a way to look through all neurons on previous layers
... now we could make them friends but it would be to much of an access for a neuron class
... but there is a solution... we can give to class neuron a pointer to the neurons on a previous layer
*/
    Layer& prevLayer = m_layers[layerNum - 1]; // very fast procedure, hust a pointer
    for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
    {
      /*
      class Neuron has its own feedForward that does 
      greedy math stuff that updates mathematical value
      */
      m_layers[layerNum][n].feedForward(prevLayer);

    }
  }
}

void Net::backProp(const vector<double>& targetVals)
{
  //calculate overall net error (RMS of output neuron errors) *RMS = "root mean square error"

  Layer& outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n)//.size-1 ... this is without bias
  {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1; //get aerage error squared
  m_error = sqrt(m_error); //RMS


  //implement a recent average measurement:(to show how well the net is trained)
  m_recentAverageError = 
    (m_recentAverageError + m_recentAverageSmoothingFactor + m_error)
    / (m_recentAverageSmoothingFactor + 1.0);

  //calculate output layer gradients 
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
  {
    //keep in mind that class Net is for loops, but Neuron is the on that actually  does the math
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  //calculate  gradients on hidden layers
  for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
  {
    Layer& hiddenLayer = m_layers[layerNum];
    Layer& nextLayer = m_layers[layerNum + 1]; //for documentation purposes
    
    for (unsigned n = 0; n < hiddenLayer.size(); ++n)
    {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }

  }

  //for all layers from outputs to first hidden layer update the connextion weights

  for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
  {
    Layer& layer = m_layers[layerNum];
    Layer& prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < layer.size(); ++n)
    {
      layer[n].updateInputWeights(prevLayer);
    }
  }
  

}

void Net::getResults(vector<double>& resultVals)/* const*/
{
  resultVals.clear();
  for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
  {
    resultVals.push_back(m_layers.back()[n].getOutputVal());
  }
}

Net::~Net()
{
}




void showVectorVals(string label, vector<double>& v)
{
  cout << label << " ";
  for (unsigned i = 0; i < v.size(); ++i) {
    cout << v[i] << " ";
  }

  cout << endl;
}




int main()
{

  //e.g., {3,2,1}, 3-2-1 = 3 inputs, 1 output
  /*
  I feel like a little bit more has to be said on the matter of 
  what is the topology vector...
  think of it as of vector with instructions for the net where each element of a vector represents 
  amount of neurons on specific layer and the whole size of topology_vector is the precise number of layers
  */

  string path  = "C:/Users/Voffka/source/repos/ConsoleApplication74/trainingData.txt";
    TrainingData trainData(path);
   // _CrtDumpMemoryLeaks();
  vector<unsigned> topology;
  trainData.getTopology(topology);

  Net myNet(topology);

  vector<double> inputVals, targetVals, resultVals;
  int trainingPass = 0;

  while (!trainData.isEof()) {
    ++trainingPass;
    cout << endl << "Pass " << trainingPass;

    // Get new input data and feed it forward:
    if (trainData.getNextInputs(inputVals) != topology[0]) {
      break;
    }
    showVectorVals(": Inputs:", inputVals);
    myNet.feedForward(inputVals);

    // Collect the net's actual output results:
    myNet.getResults(resultVals);
    showVectorVals("Outputs:", resultVals);

    // Train the net what the outputs should have been:
    trainData.getTargetOutputs(targetVals);
    showVectorVals("Targets:", targetVals);
    assert(targetVals.size() == topology.back());

    myNet.backProp(targetVals);

    // Report how well the training is working, average over recent samples:
    cout << "Net recent average error: "
      << myNet.getRecentAverageError() << endl;
  }

  cout << endl << "Done" << endl;
  //_CrtDumpMemoryLeaks();

}