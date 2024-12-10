#import libraries
import numpy as np
from scipy.stats import norm 

#variables/inputs
r = float(input("What is the risk free rate?"))
S = float(input("What is the stock price?"))
K = float(input("What is the strike price?"))
T = float(input("What is the time to maturity?"))
sigma = float(input("What is the volatility?"))
option_type = input("Enter option type ('C' for Call, 'P' for Put): ").strip().upper()

#model
def BlackScholes(r,S,K,T,sigma, type="C"):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T)) 
    d2 = d1 - sigma*np.sqrt(T) 
    try: 
        if type == "C":
            price = S*norm.cdf(d1,0,1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "P":
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Confirm all parameters please")

#output
print("The price of the option is:", round(BlackScholes(r,S,K,T,sigma, option_type), 2) )
