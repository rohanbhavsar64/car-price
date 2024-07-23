#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np


# In[19]:


import pandas as pd
transfusion = pd.read_csv("transfusion.data")


# In[20]:


transfusion.head()


# In[21]:


# Print a concise summary of the transfusion DataFrame
print(transfusion.info())


# In[24]:


# Rename the column 'whether he/she donated blood in March 2007' to 'target'
transfusion.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)

# Print the first 2 rows of the DataFrame to verify the change
print(transfusion.head(2))


# In[25]:


# Print target incidence proportions
print(transfusion['target'].value_counts(normalize=True).round(3))


# In[26]:


# Import train_test_split from sklearn.model_selection module
from sklearn.model_selection import train_test_split

# Split transfusion into X_train, X_test, y_train and y_test datasets, stratifying on the target column
X = transfusion.drop('target', axis=1)
y = transfusion['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the first 2 rows of the X_train DataFrame
print(X_train.head(2))


# In[29]:


get_ipython().system('pip install tpot')


# In[31]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[42]:


# Import TPOTClassifier from tpot and roc_auc_score from sklearn.metrics
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

# Create an instance of TPOTClassifier and assign it to tpot variable
tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, 
                       scoring='roc_auc', random_state=42, 
                       config_dict='TPOT light')

# Fit the tpot model to the training data
tpot.fit(X_train, y_train)

# Print tpot_auc_score, rounding it to 4 decimal places
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'TPOT AUC score: {tpot_auc_score:.4f}')

# Print idx and transform in the for-loop to display the pipeline steps
for idx, transform in enumerate(tpot.fitted_pipeline_):
    print(f'Step {idx}: {transform}')


# In[34]:


# Print X_train's variance using var() method and round it to 3 decimal places
print(X_train.var().round(3))


# In[37]:


# Copy X_train and X_test into X_train_normed and X_test_normed respectively
X_train_normed = X_train.copy()
X_test_normed = X_test.copy()

# Assign the column name (a string) that has the highest variance to col_to_normalize variable
col_to_normalize = X_train_normed.var().idxmax()

# For X_train and X_test DataFrames:
for df in [X_train_normed, X_test_normed]:
    # Log normalize col_to_normalize to add it to the DataFrame
    df[f'log_{col_to_normalize}'] = np.log(df[col_to_normalize])
    # Drop col_to_normalize
    df.drop(columns=[col_to_normalize], inplace=True)

# Print X_train_normed variance using var() method and round it to 3 decimal places
print(X_train_normed.var().round(3))


# In[ ]:


# Copy X_train and X_test into X_train_normed and X_test_normed respectively
X_train_normed = X_train.copy()
X_test_normed = X_test.copy()

# Assign the column name (a string) that has the highest variance to col_to_normalize variable
col_to_normalize = X_train_normed.var().idxmax()

# For X_train and X_test DataFrames:
for df in [X_train_normed, X_test_normed]:
    # Log normalize col_to_normalize to add it to the DataFrame
    df[f'log_{col_to_normalize}'] = np.log(df[col_to_normalize])
    # Drop col_to_normalize
    df.drop(columns=[col_to_normalize], inplace=True)

# Print X_train_normed variance using var() method and round it to 3 decimal places
print(X_train_normed.var().round(3))


# In[43]:


# Import linear_model from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Create an instance of linear_model.LogisticRegression and assign it to logreg variable
logreg = LogisticRegression(random_state=42)

# Train logreg model using the fit() method
logreg.fit(X_train_normed, y_train)

# Print logreg_auc_score
y_pred_proba = logreg.predict_proba(X_test_normed[:, 1])
logreg_auc_score = roc_auc_score(y_test, y_pred_proba)
print(f'Logistic Regression AUC score: {logreg_auc_score:.4f}')


# In[44]:


# Import itemgetter from operator module
from operator import itemgetter

# Create a list of (model_name, model_score) pairs
model_scores = [('TPOT', tpot_auc_score), ('Logistic Regression', logreg_auc_score)]

# Sort the list of (model_name, model_score) pairs from highest to lowest using reverse=True parameter
sorted_model_scores = sorted(model_scores, key=itemgetter(1), reverse=True)

# Print the sorted list of models
for model_name, model_score in sorted_model_scores:
    print(f'{model_name}: {model_score:.4f}')


# In[ ]:




