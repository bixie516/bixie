import numpy as np
import matplotlib.pyplot as plt
from sklearn  import linear_model
from sklearn.preprocessing import StandardScaler

data = np.array([[10,3,9,1],[9,1,7,1],[4,0,5.5,0],[6,1,8,1]])

st = StandardScaler()
data_std = st.fit_transform(data[:,:3])

lr = linear_model.LogisticRegression()
lr.fit(data_std,data[:,3])

#print(lr.coef_)

#print(lr.intercept_)

θ1 = lr.coef_[0][0]
θ2 = lr.intercept_


plt.scatter(data_std[:,0],data_std[:,1])
plt.plot(data_std[:,0],θ1*data_std[:,0]+θ2)
plt.show()
