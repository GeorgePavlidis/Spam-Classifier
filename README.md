# Spam-Classifier 
   An agent which can predict if a message is spam or not.
   
   
# Data
Input data is a set of text messages (sms), which are divided into two groups: spam and not spam with a ratio of 0.15,Only 13% of messages are classified as spam. The goal is to train the neural network so that it
can predict whether a message is spam or not.

# Learning model

This model consists of three main layers: embedding layer,convolutional, full-connected.
- The first is the embedding layer, which is responsible for the vectorization of vocabulary. The data must first be
converted to vectors in order to be used as input to the network. The conversion is performed as follows.
     * Each word is converted to an integer and created dictionary with all the words
      
  
