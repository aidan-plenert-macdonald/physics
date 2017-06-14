import re, requests, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def load_data(url):
    r = requests.get(url)
    c = r.content
    raw_data = re.findall('\<center\>(?:(?!center).|\n)*\<\/center\>', c)[1]
    month_names = map(lambda x: re.split('[<>]', x)[2],
                      re.findall('\<h2\>(?:(?!h2).|\n)*\<\/h2\>', raw_data))
    month_data  = re.findall('\<table(?:(?!table).|\n)*\<\/table\>', raw_data)
    
    tide_data = []
    for name, data in zip(month_names, month_data):
        days = re.findall('\<tr\>(?:(?!tr).|\n)*\<\/tr\>', data)[1:]
        for raw_day in days:
            day = re.split('[<>]', re.findall('\<b\>.*\<\/b\>', raw_day)[0])[2] + ', ' + name
            times = re.findall('[0-9]{1,2}:[0-9]{2}(?:(?!small).)*ft', raw_day)
            for raw_time in times:
                dt = pd.to_datetime(day + ' ' + raw_time.split('/')[0].strip())
                h  = float(raw_time.split('/')[1].replace('ft', '').strip())
                tide_data.append({'datetime':dt, 'height':h})
    return pd.DataFrame(tide_data)

df = load_data('http://tides.mobilegeographics.com/calendar/year/3220.html')
epoch = datetime.datetime.utcfromtimestamp(0)

df['seconds'] = df['datetime'].apply(lambda x: (x - epoch).total_seconds())

t, h = np.array(df['seconds']), np.array(df['height'])

def model(t, params):
    amp, freq, phase = params.reshape(3, 1, -1)
    t = t.reshape(-1, 1)
    return np.sum(amp*np.sin(freq*t + phase), axis=1)

def opt(params):
    return sum((h - model(t, params))**2)

model_terms = 15

scale = 1.0
bounds = scale*np.array([(-10, 10)]*model_terms + 
                        [(-np.pi, np.pi)]*model_terms + 
                        [(-np.pi, np.pi)]*model_terms)

print "Training"
popt = differential_evolution(opt, 
                              bounds, 
                              maxiter=10000,
                              popsize=model_terms*9)
print popt

df['predict'] = model(t, popt.x)
print df

