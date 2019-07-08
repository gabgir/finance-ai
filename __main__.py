
import neural as nl
import webdata as wd

import numpy as np
import matplotlib.pyplot as plt
# Updating the financial data.


# Loading a neural network.
nrl = nl.Neural()
nrl.load_network("./config/neural.json")


# Loading list of tickers
fd = wd.FinancialData()
fd.load_tickers("./config/ticker_technology.csv")
fd.load_compiled_data("./config/compiled_data.json")

# Test parameters
length_year = 365
num_test = 100
tick_rand = np.random.choice(fd.tickers, num_test, replace=True)

# Testing the network for average interest
interest = []
decision = []
int_reject = []
dec_reject = []
data_mem = []
for ind in range(num_test):
    try:
        tick = tick_rand[ind]
        ts = fd.create_timeseries(tick)
    except:
        continue

    if len(ts) > 2*length_year:
        # Extracting the info
        start = np.random.randint(0, len(ts)-2*length_year)
        growth = ts[start+2*length_year]/ts[start+length_year]
        data = ts[start:start+length_year]
        data = np.array(data)
        data /= data[0]

        # Applying the decision
        estimator = nrl.apply_network(vector=data)
        if np.around(estimator) == 1.:
            interest.append(growth)
            decision.append(estimator[0][0])
            data_mem.append(data)
        else:
            int_reject.append(growth)
            dec_reject.append(estimator[0][0])
    # else:
        # print("Timeseries for {} does not contain enough data".format(tick))

interest = np.array(interest)
decision = np.array(decision)
int_reject = np.array(int_reject)
dec_reject = np.array(dec_reject)

print("The average interest of accepted: {}".format(interest.mean()))
print("The average interest of rejects: {}".format(int_reject.mean()))
# print(interest)
# print(decision)

"""
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for ind in range(0, len(data_mem), 1):
    y = data_mem[ind]
    x = range(len(y))
    ax1.plot(x, y, label=interest[ind])
ax1.legend()


ax2.scatter(decision, interest)
ax2.scatter(dec_reject, int_reject)

plt.show()
"""
