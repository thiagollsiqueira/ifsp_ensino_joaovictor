#include <iostream>
#include <list>
#include <cstdlib>
#include <math.h>
#include <fstream>

using namespace std;

//exp(double x)  -> a fun��o exp pertence a biblioteca math.h, representa a opera��o: e^x
//Fun��o Logistica Sigmoid
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
//Fun��o auxiliar usada no m�todo de gradiente descendente
double dSigmoid(double x) { return x * (1 - x); }
//Fun��o utilizada para iniciar todos os pesos com n�meros entre 0 e 1
double init_weight() { return ((double)rand())/((double)RAND_MAX); }
//Fun��o para embaralhar a matriz que vai ser usada no treinamento
void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main()
{
	const int EPOCHS = 10000;
	
	// Cria��o do arquivo .CSV de sa�da para an�lise posterior dos dados
	ofstream out;
	out.open("saida.csv");
	out << "\"diferenca\",\"epoca\"\n";
	// Defini��o do Learning Rate
    const double lr = 0.1f;
    // Dimen��es da rede neural
    static const int numInputs = 2;
    static const int numHiddenNodes = 2;
    static const int numOutputs = 1;
    // Definindo as camadas escondidas e a de saida, juntamente com seus pesos e bias
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];
    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];
    
    // N�meros de treinamentos para cada �poca todos os pares de entrada possiveis (0-0, 0-1, 1-0 e 1-1)
	static const int numTrainingSets = 4;

    // Entradas
	double training_inputs[numTrainingSets][numInputs] = { {0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f} };
	// Sa�das esperadas
	double training_outputs[numTrainingSets][numOutputs] = { {0.0f},{1.0f},{1.0f},{0.0f} };

   
	// ___________Iniciando os vetores e matrizes 
    for (int i=0; i<numInputs; i++)
        for (int j=0; j<numHiddenNodes; j++)
            hiddenWeights[i][j] = init_weight();
            
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<numOutputs; j++)
            outputWeights[i][j] = init_weight();
    }

    for (int i=0; i<numOutputs; i++)
        outputLayerBias[i] = init_weight();
	//___________________________________________
	
	
    // Ordem de treinamento por �poca
    int trainingSetOrder[] = {0,1,2,3};
	
    for (int n=0; n < EPOCHS; n++) {
        // Embaralha a ordem de treino
        shuffle(trainingSetOrder,numTrainingSets);
        
        for (int x=0; x<numTrainingSets; x++) {
            int i = trainingSetOrder[x];
			
			/*
				Definindos os valores das camadas escondidas por:
				 h1 = sigmoid( ((i1 * w1) + (i2 * w2))+ bias )
				 	e
				 h2 = sigmoid( ((i1 * w1) + (i2 * w2))+ bias )
			*/
            for (int j=0; j<numHiddenNodes; j++) {
                double activation=hiddenLayerBias[j];
                for (int k=0; k<numInputs; k++)
                	activation+=training_inputs[i][k]*hiddenWeights[k][j];
                hiddenLayer[j] = sigmoid(activation);
            }
			
			/*
				Definindos o valore da camada de sa�da por:
				 o = sigmoid( ((h1 * w1) + (h2 * w2))+ bias )
			*/
            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++)
                	activation+=hiddenLayer[k]*outputWeights[k][j];
                outputLayer[j] = sigmoid(activation);
            }

			// Escrevendo no arquivo no formato: dif_saida , �poca
			// Todas diferen�as em todas as �pocas
			out << fabs(training_outputs[i][0] - outputLayer[0]) << ",";
			out << n << "\n";

			/*
				Back-propagation para atualizar os valores de pesos e bias das camadas.
				Encontrando o erro come�ando pela camada de sa�da, defindo por:
				
				 Erro = (saidaEsperada - saida) * dSigmoid(saida)
			*/
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = errorOutput*dSigmoid(outputLayer[j]);
            }
			
			/*
				Com o erro da camada de sa�da calculamos os erros da camada escondida:
				
				ErroEs1 = ((Erro * w1) + (Erro * w2)) * dSigmoid(h1)
					e
				ErroEs2 = ((Erro * w1) + (Erro * w2)) * dSigmoid(h2)
			*/
            double deltaHidden[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++)
                    errorHidden+=deltaOutput[k]*outputWeights[j][k];
                deltaHidden[j] = errorHidden*dSigmoid(hiddenLayer[j]);
            }
            /*
				Com os valores dos erros podemos alterar os pesos e bias.
				Para a camada de sa�da:
					
					Bias += Erro * lr
					W1 += h1 * Erro * lr
					W2 += h2 * Erro * lr				
			*/
            for (int j=0; j<numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++)
                    outputWeights[k][j]+=hiddenLayer[k]*deltaOutput[j]*lr;
            }
            /*
				Para as camadas escondidas:
				
				h1:									h2:
					bias += ErroEs1 * lr				bias += ErroEs2 * lr
					W1 += i1 * ErroEs1 * lr				W1 += i1 * ErroEs2 * lr
					W2 += i2 * ErroEs1 * lr				W2 += i2 * ErroEs2 * lr
			*/
            for (int j=0; j<numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                for(int k=0; k<numInputs; k++)
                    hiddenWeights[k][j]+=training_inputs[i][k]*deltaHidden[j]*lr;
            }
        }
    }
    // Escrevendo no console os valores ao final
    std::cout << "Pesos da camada escondida ([w1h1 , w2h1] , [w1h2 , w2h2])\n[ ";
    for (int j=0; j<numHiddenNodes; j++) {std::cout << "[ "; for(int k=0; k<numInputs; k++) std::cout << hiddenWeights[k][j] << " "; std::cout << "] ]"; }
	
	std::cout << "\n";

    std::cout << "Bias das camadas escondidas ([biash1 , biash2])\n[ ";
    for (int j=0; j<numHiddenNodes; j++) std::cout << hiddenLayerBias[j] << " ] ";

    std::cout << "\n";
    
    std::cout << "Pesos na camada de sa�da \n";
    for (int j=0; j<numOutputs; j++) {
        std::cout << "[ ";
        for (int k=0; k<numHiddenNodes; k++) 
            std::cout << outputWeights[k][j] << " ]";
    }
    
    std::cout << "\n";
    
    std::cout << "Bias da camada de sa�da \n[ ";
    for (int j=0; j<numOutputs; j++) std::cout << outputLayerBias[j] << " ]";
    
    std::cout << "\n";
    
    out.close();
    
    return 0;
}
