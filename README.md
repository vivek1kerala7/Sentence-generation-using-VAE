# Sentence-generation-using-VAE
This code generates a new sentence when an input sentence is provided

Modifications made on basic VAE model
Dropout layers are added at the encoder and the decoder. The way dropout layer helped in improving model is as follows: Before introducing dropout, every time the same sentence is fed to the model in a new epoch, it processed that sentence in the same manner and produced the same distribution in the latent space. This lead to model overfitting on the data and producing outputs with lesser variety. When dropout layer was added to encoder and decoder, it gave double benefit. Firstly, every time the same sentence was input, slightly different distribution was formed from that of the former case. Secondly, even if we assume that our model produced same distribution and decoder sampled same point again, still the dropout layer in decoder helped in coming up with sentences with more variety.
The KL divergence was made to decrease at a lower rate by changing the coefficient of KL divergence in loss function to a very small positive value. Due to this, the distribution that was formed didn't mimic the original distribution. Instead, it came up with distributions very similar to the original ones which helped us in achieving more variety. When we are talking about variety, do keep in mind that we did decrease reconstruction loss and KL divergence to lower values through training process and hence our model did learn language rules that are required for it to come up with new sentences.

Components of our model
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
