# Data-Science
Data Science related topics in R, Python etc.

# Machine Learning
**machine learning** - the application and science of algorithms that make sense of data
## Three different types of machine learning
1. **Supervised Learning**
   * Labeled data
   * Direct feedback
   * Predict outcome/future
2. **Unsupervised Learning**
   * No lables
   * No feedback
   * Find hidden structure in data
3. **Reinforcement learning**
   * Decision process
   * Reward system
   * Learn series of actions

### Supervised learning
Has two subcategoryes: ***classification task*** and ***regression task***
The term supervised refers to a set of samples where the desired output signals (labels) are already known.  
  
Example _e-mail spam filter_ -using a supervised machine learning algorithm on a corpus of labeled emails, emails that are
correctly marked as spam or not-spam, to predict whether a new email belongs to
either of the two categories. This belongs to classification task category.

Classification is a subcategory of supervised learning where the goal is to predict the categorical class labels of new instances, based on past observations.

A second type of supervised learning is the prediction of continuous outcomes, which is also called **regression analysis**. In regression analysis, we are given a number of predictor (explanatory) variables and a continuous response variable (outcome or target), and we try to find a relationship between those variables that allows us to predict an outcome.

Example _team budget and standing at the end of seson_ - the more money you have the better place you will get

### Reinforcement learning
In reinforcement learning, the goal is to develop a system (agent) that improves its performance based on interactions with the environment. Since the information about the current state of the environment typically also includes a so-called reward signal, we can think of reinforcement learning as a field related to supervised learning.  
Example _chess match_ - Here, the agent decides upon a series of moves depending on the state of the board (the environment), and the reward can be defined as win or lose at the end of the game.
