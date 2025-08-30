# 📱 SMS Spam Classification in R (caret + quanteda)

Classify SMS messages as **ham** or **spam** using a traditional ML workflow in R: text preprocessing with **quanteda**, a **Document-Feature Matrix (DFM)**, and a **Decision Tree (rpart via caret)** with repeated cross-validation.
![Distribution of Text Lengths with Class Labels] <img width="575" height="304" alt="Rplot" src="https://github.com/user-attachments/assets/8ffc5a9b-b1a7-4488-8084-018e2c8fada8" />

---

## 🔎 Project Overview

- **Goal:** Build a baseline, interpretable classifier for SMS spam detection.
- **Pipeline:** Clean → tokenize → DFM → 70/30 split → Train Decision Tree (CV) → Evaluate.

---

## 🔧 Code Overview (brief)

### Libraries
- **quanteda** – Fast text preprocessing (tokenize, lowercase, remove stopwords, stem) and DFM creation to turn raw text into numeric features.
- **caret** – Unified model training & evaluation (`train()`, `trainControl()`, `createDataPartition()`, `confusionMatrix()`).
- **rpart** (via caret) – Decision Tree as a quick, interpretable **baseline**.
- **ggplot2** – EDA plot to visualize ham vs spam message lengths.

### Preprocessing
- `tokens()` + `tokens_tolower()` + `tokens_select(stopwords)` + `tokens_wordstem()` → Normalize and reduce noise/sparsity.
- `dfm()` → Bag-of-words features (rows=messages, cols=terms).
- `dfm_match(test, features=featnames(train))` → Align test columns to the **training vocabulary**.
- `convert(dfm, to="data.frame")` + `make.names()` → Modeling-ready data frame with valid column names.

### Modeling & Evaluation
- `createDataPartition(p=0.7)` → 70/30, **stratified** split (keeps ham/spam ratio stable).
- `train(method="rpart", tuneLength=7)` with `trainControl(repeatedcv, number=10, repeats=3)` → CV for a stable estimate and simple hyperparam search.
- `confusionMatrix()` → Accuracy, Kappa, Sensitivity (ham), Specificity (spam), Balanced Accuracy.

---

## 📊 Your EDA & Results

### Class distribution
```
 ham spam 
0.86 0.14 
```

### Text length (characters) summary
**Ham**
```
Min. 1st Qu. Median  Mean 3rd Qu.  Max. 
2.00   33.00  53.00 72.29  96.00  910.00 
```
**Spam**
```
Min. 1st Qu. Median  Mean 3rd Qu.  Max. 
33.0  132.8  148.0  139.2 156.0   223.0 
```

### Training confusion matrix
```
Confusion Matrix and Statistics

           
trainresult  ham spam
       ham  1149   49
       spam   55  147
                                          
               Accuracy : 0.9257          
                 95% CI : (0.9107, 0.9389)
    No Information Rate : 0.86            
    P-Value [Acc > NIR] : 1.198e-14       
                                          
                  Kappa : 0.6954          
                                          
 Mcnemar's Test P-Value : 0.6239          
                                          
            Sensitivity : 0.9543          
            Specificity : 0.7500          
         Pos Pred Value : 0.9591          
         Neg Pred Value : 0.7277          
             Prevalence : 0.8600          
         Detection Rate : 0.8207          
   Detection Prevalence : 0.8557          
      Balanced Accuracy : 0.8522          
                                          
       'Positive' Class : ham             
```

### Test confusion matrix
```
Confusion Matrix and Statistics

          
testresult ham spam
      ham  479   27
      spam  37   57
                                          
               Accuracy : 0.8933          
                 95% CI : (0.8658, 0.9169)
    No Information Rate : 0.86            
    P-Value [Acc > NIR] : 0.009078        
                                          
                  Kappa : 0.5781          
                                          
 Mcnemar's Test P-Value : 0.260589        
                                          
            Sensitivity : 0.9283          
            Specificity : 0.6786          
         Pos Pred Value : 0.9466          
         Neg Pred Value : 0.6064          
             Prevalence : 0.8600          
         Detection Rate : 0.7983          
   Detection Prevalence : 0.8433          
      Balanced Accuracy : 0.8034          
                                          
       'Positive' Class : ham             
```
