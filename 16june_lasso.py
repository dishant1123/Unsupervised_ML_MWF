"""
Lasso Regression is Linear Regression + penalty on absolute values of coefficients. It shrinks less important feature coefficients to exactly zero, removing them from the model.

Cost = RSS + λ × Σ |βj|

• RSS = Residual Sum of Squares
• λ (lambda) = Regularization strength
• β = Feature coefficients → larger λ pushes more β to 0

use : 
1. many  features some irrelevant
2.feature >samples   : like  100 patients 1000 medical documents 
3. overfitting of training  data  ===> model  work on train well  but test fail  

advantages :
1. automatically  select  best  features
2.reduce overfitting ,simplify model,better generalization

disadvantages :
1. sometime  remove  important  feature
2. can underfit  model : very large  lambda then remove many features 

alpha controls the strength of regularization.
Mathematically:

alpha = λ (lambda)
Higher alpha ⇒ stronger penalty.
Lower alpha ⇒ weaker penalty.

"""

import pandas as pd
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "area" : [1000,1200,1500,2000,2500],
    "bedrooms":[2,2,3,3,4],
    "age" :[15,18,20,25,30],
    "parking_slot":[1,1,1,2,2]
})
x=pd.DataFrame(data)
y = [30,35,40,48,57]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = Lasso(alpha=1)
model.fit(x_scaled,y)

print("coefs :",model.coef_)   # coefs : 
print("R2score : ",r2_score(y,model.predict(x_scaled))) #