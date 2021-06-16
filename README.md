# Sentence-generation-using-VAE
This code generates a new sentence when an input sentence is provided

Modifications made on basic VAE model
Dropout layers are added at the encoder and the decoder. The way dropout layer helped in improving model is as follows: Before introducing dropout, every time the same sentence is fed to the model in a new epoch, it processed that sentence in the same manner and produced the same distribution in the latent space. This lead to model overfitting on the data and producing outputs with lesser variety. When dropout layer was added to encoder and decoder, it gave double benefit. Firstly, every time the same sentence was input, slightly different distribution was formed from that of the former case. Secondly, even if we assume that our model produced same distribution and decoder sampled same point again, still the dropout layer in decoder helped in coming up with sentences with more variety.
The KL divergence was made to decrease at a lower rate by changing the coefficient of KL divergence in loss function to a very small positive value. Due to this, the distribution that was formed didn't mimic the original distribution. Instead, it came up with distributions very similar to the original ones which helped us in achieving more variety. When we are talking about variety, do keep in mind that we did decrease reconstruction loss and KL divergence to lower values through training process and hence our model did learn language rules that are required for it to come up with new sentences.

**Components of our model**
1. Embedding layer
This layer takes the input and converts it into corresponding word embedding. We have chosen 300 dimensional word embedding. This means the input words will be converted to 300 dimensional embeddings.
2. Encoder
Encoder takes the embedded input and converts it into an encoded representation. We have chosen a bidirectional LSTM encoder with 2 hidden layers each of size 256.
3. Latent space
Latent space is where the Gaussian distribution of latent variable is formed. From the output of encoder, we will find the mean and variance of the Gaussian distribution of corresponding latent variable. 
4. Decoder
Decoder samples a point from this distribution and produces an output. We are using a unidirectional LSTM decoder with 4 hidden layers of size 256.
5. Dropout layer
Dropout layer is used to bring in some uncertainty in the network. We have set probability of dropout as 0.3. This means, every neuron in the dropout layer may get dysfunctional with a probability of 0.3. This will restrict our network from overfitting by learning exact patterns of training data.

**The approach used**
Our input is a batch of sentences. VAE will take it one by one and pass through the embedding layer. We will initialize the embedding layer with the vocabulary that we had built using Glove. The embedding layer can now create an embedded vector for each word which can be passed to the encoder. 
Encoder takes the output of embedding layer as input and produces the hidden state output. Next task is to find out the Gaussian distribution of latent variable from the encoder's hidden state. A Gaussian distribution is characterized by the mean and variance of that distribution. We have to find the mean and variance from the hidden state output produced by the encoder. This can be found by passing the hidden state output through a linear layer to produce mean and variance vector of the size of dimension of latent variable. These linear layers will learn to produce mean and variance when we train our model.
Once we have mean and variance, we have found the Gaussian distribution corresponding to the latent variable. This is an approximation of the distribution of latent variable found by us. We will find the KL divergence of this distribution from the original distribution of the latent variable which will tell us how close is our distribution to the original distribution. 
Now, a random point is sampled from this distribution and fed into the decoder. Decoder takes this input and produces output. This output will have vector representation of every single word in every sentence. Now, we have to identify words from this vector representation and compare with the initial input given to the encoder. 
In training phase, we train so that the VAE should be able to produce the exact sentence provided at input as the final output. The error in achieving this is called as reconstruction error. We will use cross entropy error as the error function. When we provide decoder output and original encoder input to cross entropy function, it will return us the cross entropy error which is a measure of how well we were able to reconstruct the sentence.
Our loss function is not complete yet. There is one more contributor to our loss function. It is the KL divergence that we had seen earlier. Our aim is to decrease the reconstruction loss as well as the KL divergence. If we simply look at the reconstruction loss, out model may simply start replicating given inputs. But with the KL divergence term, it will be able to come up with new sentences. Our total loss is the sum of reconstruction loss and KL divergence. 
Here we are finding the KL divergence between q(z/x) and p(z/x) where z is our latent variable and q(z/x) is approximation of p(z/x). It is difficult for us to find p(z/x) directly as p(z/x) = p(x/z)p(z)/p(x) and p(x) is computationally infeasible. Hence we approximate it with q(z/x) and try to keep the KL divergence between q(z/x) and p(z/x) to the minimum so that q(z/x) remains as a good approximation of p(z/x).

In the above loss function, a and b denote coefficients of reconstruction loss and KL divergence respectively. In our implementation, we will use cross entropy loss to find reconstruction loss. We have used coefficient of reconstruction loss as 8 and coefficient of KL divergence as 0.001.
This error is backpropagated in our neural network in order to calculate gradient of error with respect to each of the network weights and calculate the updated weights. This backpropagation and weight updation is done at the end of each batch. We are using a batch size of 128. This means weights will be updated whenever inspection of a batch of 128 sentences is completed. For optimization, we are using Adam optimizer with a learning rate of 0.0001.


