import numpy as np
import matplotlib.pyplot as plt
class LinearRegressor:
    def __init__(self,learningrate,epochs):
        self.w=None
        self.b=None
        self.learningrate=learningrate
        self.epochs=epochs
    

    def predict(self,xt):
        m=xt.shape[0]
        n=xt.shape[1]
        ypreds=[0 for _ in range(m)]
        for i in range(m):
            ypredi=0
            for k in range(n):
                ypredi+=self.w[k]*xt[i][k]
            ypredi+=self.b
            ypreds[i]=ypredi   
        return ypreds


    def fit(self,x,y):
        
        m=x.shape[0]
        n=x.shape[1]
        self.w=np.random.randn(n)
        self.b=0
        for epoch in range(self.epochs):
            dw=[0 for _ in range(n)]
            db=0
            for i in range(m):
                ypredi=0
                for k in range(n):
                    ypredi+=self.w[k]*x[i][k]
                ypredi+=self.b  
                error=abs(y[i]-ypredi)
                for k in range(n):
                    dw[k]+=2*error*x[i][k]
                db+=2*error
            for k in range(n):
                dw[k]=dw[k]/m
            db/=m
            for k in range(n):
                self.w[k]-=self.learningrate*dw[k]
            self.b-=self.learningrate*db     


np.random.seed(42)
m=100
X=np.random.randn(m,2)
X = (X - X.mean(axis=0)) / X.std(axis=0)
truew=np.array([3,5])
trueb=7
noise=np.random.randn(m)*0.5
y=X @ truew +trueb +noise 
model=LinearRegressor(0.001,100)
model.fit(X,y)
n=25
Xtest=np.random.randn(n,2)
Xtest = (Xtest - Xtest.mean(axis=0)) / Xtest.std(axis=0)
truewt=np.array([3,5])
truebt=7
noise=np.random.randn(n)*0.5
ytest=Xtest @ truewt +truebt +noise 
predictions=model.predict(Xtest)
print(predictions)
error=0
for i in range(n):
    error+=abs(ytest[i]-predictions[i])
error/=n
print(f"the MAE is {error:.3f}")  
plt.scatter(range(n),ytest,label='True')
plt.scatter(range(n),predictions,label='Predicted',marker='x')
plt.legend()
plt.title("True vs Predicted")
plt.show()

        
