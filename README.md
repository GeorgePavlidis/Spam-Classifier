# Spam-Classifier 
   An agent which can predict if a message is spam or not.
   
   
# Data
Input data is a set of text messages (sms), which are divided into two groups: spam and not spam with a ratio of 0.15,Only 13% of messages are classified as spam. The goal is to train the neural network so that it
can predict whether a message is spam or not.

# Learning model

This model consists of three main layers: embedding layer,convolutional, full-connected.
- The first is the embedding layer, which is responsible for the vectorization of vocabulary. The data must first be
converted to vectors in order to be used as input to the network. The conversion is performed as follows:
     * Each word is converted to an integer and created dictionary with all the words
     * And then the embedding layer finds semantics relationships between words, places words with a similar meaning
     closer than what to words with a different one meaning.
   Το κάθε στοιχείο από το σύνολο των δεδομένων μετατρέπετε σε ένα διάνυσμα μήκους 188. Το μήκος αυτό προκύπτει το
   μήκος του μεγαλύτερου μηνύματος, η μετατροπή αυτή γίνεται με σκοπό όλα τα διανύσματα εισόδου να έχουν το ίδιο
   μέγεθος. Στα μηνύματα με αρχικά μικρότερο μήκος, προστίθενται μηδενικά στο τέλος του διανυσμάτων τους μέχρι να
   είναι ίσο 188 το μήκος τους. Αντίστοιχαδιανυσματοποιούνται και τα labels όπου το [1,0] είναι το σπαμ και
   το [0,1] είναι κανονικό μήνυμα.

      
  
