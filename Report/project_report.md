# Snorkel Project Report

## Motivation: 
	 The overarching goal here is to mine biological literature and extract relevant know biological relationships e.g. Disease-Gene relationships. Ultimately this project will effectively expand the amount of relationships that are currently referenced in project [rephetio] (https://thinklab.com/p/rephetio). 

## Methods
### How Snorkel Works
	To accomplish the goals that were laid out above, we decided to use [snorkel] (https://github.com/HazyResearch/snorkel), which is a lightweight natural language processor (nlp). The idea behind snorkel is that  one doesn’t have to look and categorize every potential candidate relationship, but instead write simple pythonic functions (aka noisy label functions) that will accomplish this task. An example of a label function is provided in the figure below.

	```python
	def LF_KB (c):
		if c[0].mesh_id in KB and c[1].mesh_id in KB:
			return 1
		return -1
	```

	In this paraphrased example above, this label function will assess if a disease's [meshid] (https://meshb.nlm.nih.gov/#/fieldSearch) is contained in our knowledge base. The 1 or -1 corresponds to the positive and negative candidate labels respectively; furthermore, there is also the possibility of a potential candidate receiving a 0, which means a particular label function can’t tell if a candidate is real (+) or fake (-). Overall, the goal for these label functions is to label as many candidates as possible. Now warning to this statement is if a label function spits out non-sense labels, then the model might not work well. After the labeling process, the next step is to feed the generated label matrix into a generative model, e.g. naïve bayes, to model a consensus of all the label functions with respect to each candidate. Once training the generative model has finished, the next step is to load the training marginals (prior of each candidate belonging to positive class) into a discriminator model (e.g. logistic regression) along with generated nlp features. Ultimately, the discriminator will generate probabilities of the test candidates will belong to either the positive class (> 0.5) or the negative class (< 0.5).  

### Epilepsy Trial Run
	To assess the power of snorkel we designed a proof of concept experiment, where we grab all epilepsy related pubmed abstracts from [pubtator] (https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/PubTator/) and determined to see if we can classify Disease-Gene entity relationships that are already in our knowledge base. After gathering and parsing the abstracts, we stratified the data into three equal groups (train, validation, and test). Using these three different groups and a combined total of 16 label functions, we trained a naive bayes (generative model) for gathering our labeling consensus and trained a logistic regression (discriminator model) to predict our test set candidates. These 16 label functions consist of whether a candidate is in our knowledge base (KB) and further context specific annotations (KB+Context). 

## Results:
	Out of ~6,000 candidates, we reached a AUROC of 0.61 and 0.58 from our KB and KB+Context labeling models respectfully. ![ROC Curve][roc] These results suggest that adding context label functions performs just as well as solely using our knowledge base labels. Digging further into our analysis, we identified the most prominent feature from the KB model which turned out to be if a candidate had the word epilepsy to the right of a mention. Feature weights are located in data/features.tsv and the distribution of weights can be seen here. ![Feature Weights][feat-weights] The feature weight distribution is cenetered around zeros, which suggests this model can remove uninformative features. Following these weights, we plotted a probability correlation plot that shows both models are able to identify negative candidates (shown by the dark regions). ![Hexbins][hex] Overall, one potential downfall for this experiment is that we are dealing with a small sample size (20 thousand documents compared to the full 2 million).

## Future Directions:
	All in all, this framework shows promise in finding Disease-Gene relationships. Since epilepsy is a small subset compared to the full pubmed dataset, there is an issue of a small sample size. Obviously, the next step if this project were to be continued is to utilize all the available abstracts at once. Furthermore, having outside expertise in identifying which terms are diseases or disease synonyms compared to just symptoms of a disease, would help validate our current set of label functions. Another direction this project will go is to explore other biological relationships such as Compound-Disease, Gene-Gene, Gene-SNP, Species-Disease, etc. or possibly consider a tertiary level of candidates (Disease-Gene-SNP etc.). Lastly, it be interesting to see how this framework performs with different machine learning approaches. So far I used logistic regression and naïve bayes, but other deep learning models might have an increase in predictive power. Possible deep learning algorithms would be Generative Adversarial Networks (GAN) or some type of neural network (e.g. Long short term memory(LSTM)) can be considered. 

[roc]: roc-curve.png "ROC Curve KB vs KB_Context"
[feat-weights]: feature-weight-KB.png "Feature Weight (KB)"
[hex]: hexbin-prediction-correlation.png "Hexbin Correlation Plot" 