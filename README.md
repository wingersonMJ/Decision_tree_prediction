
# A clinically intuitive approach to predicting persisting symptoms after concussion in adolescent specialty care

**Authors:** Mathew J. Wingerson, MS,(1,2) Joshua R. Kniss, PT, DPT,(1,2) Julie C. Wilson, MD,(1,2,3) David R. Howell, PhD, ATC,(1,2)  

**Author affiliations:**  
1 Department of Orthopedics, University of Colorado School of Medicine, Aurora, CO, USA
2 Sports Medicine Center, Children’s Hospital of Colorado, Aurora, CO, USA
3 Department of Pediatrics, University of Colorado School of Medicine, Aurora, CO, USA

***Currently under journal review - citation and PubMed link forthcoming!***  
*Data available upon reasonable request*  

## ABSTRACT

**Background:** Initial post-concussion symptoms and the number of days between injury and symptom reporting are important factors to consider when 
prognosing recovery time in adolescents with recent concussion. We sought to explore interactions between these two factors in predicting patient odds of 
developing Persisting Symptoms after Concussion (PSaC), and to determine if initial post-injury symptom scores and time to evaluation could differentiate 
between patients who did and did not develop PSaC in a decision tree framework.  

**Methods:** We reviewed medical charts for patients 8-18 years old who presented for post-concussion care within 21-days of injury, and extracted 
self-reported symptoms on the Health and Behavior Inventory (HBI) and the time (in days) between injury and clinical evaluation. Logistic regression 
explored interactions between HBI score and time to evaluation. Then, a decision tree was used to create clinically meaningful cut points in HBI score and 
time to evaluation for predicting PSaC development.  

**Results:** 305 patients were evaluated (14.5±2.3 years old; 37.4% female; 8.7±5.4 days post-concussion); N=60 (19.7%) developed PSaC and N=245 (80.3%) 
did not. After controlling for covariates of sex, age, and concussion history, there was a statistically significant interaction between HBI score and 
time to evaluation (Odds Ratio: 1.01; 95% Confidence Interval: 1.00, 1.01; p-value = 0.049), such that the effect of HBI score on odds of PSaC development 
was dependent on the number of days between injury and symptom reporting. Our decision tree using HBI score and time to evaluation correctly classified 81.
9% of patients, with an AUC of 0.71 and a sensitivity and specificity of 26.7% and 95.4%, respectively.  

**Conclusions:** Stratifying patients based on just two factors – self-reported symptoms and time to evaluation – can provide an estimate of PSaC risk, 
though cannot replace more comprehensive prediction tools, clinical judgement, or the standard multifaceted evaluation approach recommended by recent 
consensus statements. As a rule-of-thumb guide supporting existing clinical reasoning, and for use in time- or resource-limited clinical settings, our 
decision tree contextualizes how the timing of symptom assessment, in addition to symptom burden, relates to recovery in adolescents with concussion.  

## Abreviated Methods (see full manuscript for complete methods)

#### Statistical Analysis:

Demographics and injury characteristics are reported as mean (standard deviation) or number in group (%). For our primary purpose, we used multivariable 
logistic regression to determine if time to evaluation and HBI score were significant predictors of odds of PSaC development. We then added an ‘HBI X time 
to evaluation’ interaction term to the model to determine if the effect of HBI score on PSaC odds was dependent on the number of days that had elapsed 
between injury and the evaluation. Last, we evaluated the final interaction model with the addition of demographic covariates known to affect PSaC risk, 
which were determined a priori and included age, biological sex, and concussion history. A likelihood ratio test compared goodness of fit 
between models. Significance was set at p<0.05 and all tests were two sided.  

#### Building the Decision Tree:

To address our secondary aim of developing a simple, intuitive framework for understanding how symptom reporting and time to evaluation combine to affect 
PSaC development, we constructed a single decision tree using HBI score and time to evaluation as features for splitting nodes. Split decisions for each 
node were determined using Gini impurity scores, where a lower score for a potential split indicated greater homogeneity of PSaC vs no PSaC patients in 
the resulting nodes (e.g., a better split). Pre-pruning prevented terminal nodes (leaves) from having fewer than 5 samples to reduce overfitting to 
individual cases. Minimal cost-complexity post-pruning (CCP) was used to reduce the final decision tree size and lower the risk of overfitting to small 
data sets or failing to generalize to unseen data. CCP is a post-pruning technique that creates several subtrees, reduces the subtree complexity (number 
of splits), re-evaluates misclassification rates, then selects the subtree that balances complexity and accuracy. The CCP-alpha term, which controls the 
complexity (number of splits) in the subtree, was selected using cross-validation. The CCP-alpha value with the highest classification accuracy in 
cross-validation was used in the final decision tree. Results for the terminal nodes of the final decision tree are presented as follows: number of total 
samples in the leaf, number of samples with PSaC (and %), class prediction (PSaC or no PSaC), and relative PSaC risk. Relative PSaC risk is the calculated 
probability of having PSaC in a leaf compared to the probability of having PSaC in the entire cohort. A relative risk ratio of 1.0 indicates that 
patients in a particular leaf are exactly as likely to have PSaC as the entire study sample. A relative risk ratio of <1.0 indicates a reduced risk of 
PSaC compared to the remaining sample, and >1.0 indicates an increased risk for the patients in that leaf.  

#### Tree Evaluation/Cross-Validation: 

In the absence of a separate cohort for evaluation of performance metrics for the developed decision tree, we used stratified k-fold cross-validation. The 
number of folds was selected in accordance with leave-one-out principles, which is known to provide more accurate estimates of model generalization among 
small sample sizes. K was set at the number of samples in the minority class. Performance metrics reported were Receiver Operating Characteristic 
Area Under the Curve (AUC), overall classification accuracy (correct classifications / all classification attempts * 100[%]), and sensitivity/specificity. 

Included in the supplementary materials, we reported bootstrap estimates of the mean and 95% confidence interval for the percentage of patients with PSaC 
and the relative PSaC risk in each leaf. Bootstrapping was performed with replacement, but we did not re-build the tree for each bootstrap; instead, we 
estimated percentage of patients with PSaC and relative PSaC risk at each leaf using the final tree structure developed from the full dataset. Our 
objective in bootstrapping was to quantify the uncertainty of these estimates and provide context for how observed percentages and risk ratios may vary in 
other samples. We did not stratify the resampling by PSaC class, primarily because the overall prevalence of PSaC varies substantially across the 
literature. Our sample was recruited from a specialty care clinic and its PSaC prevalence may not mirror that of other patient populations. By allowing 
for flexibility in the PSaC proportion in each bootstrapped sample, our confidence intervals are expected to be wider (i.e., less certain) but more 
generalizable and representative of a range of values that might be observed in other samples with higher or lower PSaC prevalence. 

Data analysis was conducted in Python programing language (Python Software Foundation,  https://www.python.org/). Scripts are available at *will add link 
when I upload it to Github*.  

## Results

#### Sample description: 

A total of 558 patients with diagnosed concussion were included in the study, of which 350 were followed longitudinally until symptom resolution. After 
removing patients with an HBI <1 or who did not have data for days between injury and evaluation, the final sample for analysis was 305 patients (Table 
1). N=60 patients (19.7%) developed PSaC, and the remaining N=245 (80.3%) did not.  

***Table 1.** Demographics and injury characteristics for the sample (N=305). Data are described as mean (standard deviation) for continuous variables and 
N (% in group) for categorical.*
| Variable                | Mean (sd) or N (%) |
| ----------------------- | ------------------ |
| Age (years)             | 14.5 (2.3)         |
| Biological sex (female) | 114 (37.4%)        |
| History of concussion   | 128 (41.9%)        |
| ADD/ADHD diagnosis      | 26 (8.5%)          |
| History of anxiety      | 24 (7.9%)          |
| History of depression	  | 13 (4.3%)          |
| Time to symptom resolution (days) | 21.8 (24.0) |  
<br>

**Table 1.2.** Continuation of table 1...
| Variable of interest       |  Mean (sd) or N (%) | Range  |
| -------------------------- | ------------------- | ------ | 
| Time since injury (days)	 | 8.7 (5.4)           | 1 – 21 |
| HBI total score            | 19.5 (12.7)         | 1 – 58 |
| --> HBI cognitive subscale | 10.6 (8.0)          | 0 – 31 |
| --> HBI somatic subscale   | 8.9 (5.8)           | 0 – 27 |
<br>


#### Regression models:

In a multivariable logistic regression with HBI score and time to evaluation as predictors, a higher HBI score at the initial post-concussion visit and a 
longer time to evaluation were statistically significantly associated with higher PSaC odds (Table 2A). The addition of an interaction term between HBI 
score and time to evaluation did not meet statistical significance (Table 2B); however, after controlling for covariates of sex, age, and concussion 
history, there was a statistically significant interaction between HBI and time since injury (Table 2C), such that the effect of HBI score on odds of PSaC 
development was dependent on the number of days between injury and evaluation. The likelihood ratio test between the interaction model with covariates and 
the first multivariable logistic regression was statistically significant (p=0.027), therefore the more complex model is a better fit to the data despite 
including more variables.  

**Table 2.** Multivariable logistic regression results. Data are presented as odds ratio for predicting PSaC, upper and lower 95% confidence intervals for the odds ratio, and p-value. 
| Variable                               | Odds Ratio | 95% CI: Upper | 95% CI: Lower | P-value|
| :------------------------------------- | :--------: | :-----------: | :-----------: | -----: |
| ***A. Multivariable model***           |            |               |               |        |
| HBI score                              |1.08        | 1.05          | 1.11          | <0.001 |
| Time to evaluation                     | 1.17       | 1.10          | 1.25          | <0.001 |
| ***B. Multivariable model + Interaction term*** |   |               |               |        |
| HBI score                              | 1.03       | 0.97          | 1.09          | 0.290 |
| Time to evaluation                     |1.05        | 0.92          | 1.20          | 0.441 |
| HBI x Time to evaluation (interaction) | 1.00       | 0.99          | 1.01          | 0.067 |
| ***C. Multivariable model + Interaction term + covariates*** | |    |               |       |
| HBI score	                             | 1.03       | 0.97          | 1.09          | 0.340 |
| Time to evaluation                     | 1.05       | 0.92          | 1.20          | 0.492 |
| HBI x Time to evaluation (interaction) | 1.01       | 1.00          | 1.01          | 0.049 |
| Age (years)                            | 0.93       | 0.80          | 1.08          | 0.354 |
| Sex (female)                           | 1.79       | 0.90          | 3.54          | 0.095 |
| Concussion history (yes)               | 2.00       | 1.01          | 3.97          | 0.048 |
<br>

Our decision tree used HBI score and time to evaluation at each split point. CCP is demonstrated in the Supplementary File. The final tree contained five 
terminal nodes/leaves and is presented in Figure 1. Number of samples, percent PSaC, and relative PSaC risk for each leaf are reported in Figure 1 and are 
in reference to our full sample of 305 subjects. Mean and 95% Confidence Intervals for percentages and risk ratios, as derived from bootstrapping, are 
included in the Supplementary File. The decision boundary for the tree is presented in Figure 2. Across 60 instances of stratified cross-validation, the 
final decision tree had a mean classification accuracy of 81.9%, a mean AUC of 0.7096, and a sensitivity and specificity of 26.7% and 95.4%,
respectively.  

**Figure 1.** The final decision tree using initial post-concussion HBI score and time to evaluation as split points. Blue-leafs are predicted as not 
having PSaC; orange leafs are predicted as having PSaC. Number of samples, percent PSaC, and relative PSaC risk for each leaf are reported and are in 
reference to our full sample of 305 subjects. Mean and 95% Confidence Intervals for percentages and risk ratios, as derived from bootstrapping, are 
included in the Supplementary File.  
<img src="figs\official_tree.jpg" alt="Decision Tree" width="600">  
*Footnote.* Risk ratios are calculated with respect to the overall samples’ prevalence of PSaC (i.e., 20% with PSaC). Therefore, a risk ratio of 1.0 
indicates that the risk of PSaC in that leaf is equal to the risk of PSaC in the sample as a whole. A risk ratio &lt;1.0 indicates a lower risk compared 
to the remaining sample. A risk ratio &gt;1.0 indicates a greater risk.  
<br> 

**Figure 2.** Decision boundary for predicting PSaC using HBI scores and time to evaluation. Blue squares are patients without PSaC, orange triangles are 
patients with PSaC. The shaded orange region are values of HBI scores and time to evaluation where the decision tree would predict PSaC development. The 
blue regions, the tree would predict no PSaC development. 
<img src="figs\decision_boundary.png" alt="Decision Boundary" width="600">  


## Conclusion 

We created a simple framework for using symptom burden and time to evaluation to predict PSaC development in adolescents with concussion. Stratifying 
patients based on these two factors can provide an estimate of relative PSaC risk, though cannot replace more comprehensive tools (e.g., 5P), clinical 
judgement, or the standard multifaceted evaluation approach recommended by recent consensus statements. As a rule-of-thumb guide supporting existing 
clinical reasoning, our decision tree contextualizes how the timing of symptom assessment, in addition to symptom burden, relates to recovery in 
adolescents with concussion. 


## Supplemental Figures 

### Understanding parameter selecction:

**Supplementary Figure 1.** Selection of Cost-Complexity post-pruning (CCP)-alpha based on accuracy during cross-validation. The CCP-a value with the 
highest validation set accuracy was used for tree building. This maximizes potential generalization to unseen data and reduces risk of overfitting. 

<img src="figs\Figure_1.png" alt="accuracy vs alpha" width="600">  

**Footnote.**  
*Understanding CCP-Alpha.* The x-axis represents potential values of alpha that could be used during Cost-Complexity Post-Pruning (CCP). The objective of CCP is to reduce the size of the decision tree, thereby promoting generalizability to unseen data by balancing the overall accuracy of the decision tree with the tree depth (i.e., complexity). An unpruned decision tree (alpha 0.000 on the x-axis) resulted in a tree with significant depth and size (e.g., several more splits than our final tree). Performance in the training data, where the tree was developed, was high: approximately 95% of patients were correctly classified. However, the highly complex, non-pruned tree performed poorly on unseen data – a classic sign of overfitting – with an accuracy of approximately 75%.  

*Selection of CCP-Alpha.* We selected a CCP-alpha value that maximized performance in both the training and validation datasets, around alpha=0.015. 
<br>
<br>

### Bootstrapping for confidence intervals: 

**Supplementary Figure 2A.** Bootstrapped means and confidence intervals for the percentage of patients with PSaC and relative PSaC risk in each terminal 
leaf. Bootstrapping was conducted with replacement and for 1,000 iterations. 

<img src="figs\official_tree_confidence_intervals.jpg" alt="Bootstrapped decision tree" width="600">  

*Footnote.* Risk ratios are calculated with respect to the overall bootstrapped samples prevalence of PSaC, which varied in each iteration because of our 
replacement sampling. A risk ratio of 1.0 indicates that the risk of PSaC in that leaf is equal to the probability of PSaC in the sample as a whole. A 
risk ratio <1.0 indicates a lower risk compared to the remaining sample. A risk ratio >1.0 indicates a greater risk. 
<br>
<br>

### Clinically-relevant take-away: 

**Supplementary Figure 2B.** A fun version of the same info as above, maybe good for a poster. This includes only the 95% Confidence Intervals for the 
relative PSaC risk ratios in bootstrapping, mostly because I felt that would be most relevant to an ‘outside’ person wanting to use this in their own 
clinic. Bootstrapping gives us our closest estimate for how this could look in an outside sample, so that is the info included here.  

<img src="figs\hbi_time_riskTree_confidence_intervals.jpg" alt="Bootstrapped decision tree" width="600">  

*Footnote.* Interpretation remains the same. For the far left leaf, risk is estimated to be the lowest among all the leaves, so these kids have 0.9 to 0.
45 the risk of PSaC compared to the rest of the sample (or, less than half as likely, we think). For the far right leaf, risk is estimated to be much 
higher, 3-5 times higher, than what we would expect in the ‘average’ patient. 