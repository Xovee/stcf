# Code

The structure of the codes is as follows:

- For pretraining: `sc_pretrain.py` and `tc_pretrain.py`
- For fine-tuning: `stcf.py`
- For evaluation: `evaluate.py`
- Metrics are defined in `metrics.py`
- Some utility functions are defind in `utils.py`

## Note on Evaluation Metrics

In the paper, we defined five metrics. Since the objective function of the model is MSE, we used MSE mainly for most of the experiments described in the paper.  These five metrics behaved differently -- depending on the practical requirement, you should 
choose the most appropriate one(s):

- Mean Squared Error (MSE): This is one of the most common metrics for regression task. It gives more weight to larger errors. That is to say, our model is more focused on inferring traffic flows on regions with higher flows. In real-world, we are often more concerned with heavy traffic regions. Therefore, we choose MSE as the main metric in our paper.  
- Mean Absolute Error (MAE): This metric treats all errors equally -- whether big or small. It also has a preference for regions with higher flow volumes.  
- Mean Absolute Percentage Error (MAPE): This metric uses the percentage form of the errors. This metric is suitable for situations that the prediction values are scaled from a large spectrum -- dozens to tens of thousands, making it scale-independent. MAPE has a notable limitation: it can be sensitive when the values are close to zero or even undefined for values of zero.  
- Mean Squared Logarithmic Error (MSLE): Useful for situations that the distribution of the traffic flows is skewed -- less sensitive to large errors. It penalizes underestimated predictions, i.e., a slight overestimation is more preferred.
- Accuracy With K% Tolerance (ACC@K%): This is a classification metric. The prediction for a single cell would be classified as correct once the absolute percentage error is less than K%. Use this metric when you want to have a feeling about that given a threshold, how many cells (and where) are correctly predicted. However, this metric ignores the distance between predictions and ground-truth, it should be only used as a supplementary metric. 