from flask import Flask, request, jsonify
import math
import scipy.stats
import statistics
from scipy.stats import t
from scipy.stats import norm
from typing import List
import numpy as np
import sympy as sp

def Factorial(num: int) -> int:
    ans = 1
    while num != 0:
        ans *= num
        num -= 1
    
    return ans

def Combination(n: int, x: int) -> float:
    ans = Factorial(n)
    temp = Factorial(n - x)
    temp *= Factorial(x)
    ans /= temp
    ans = round(ans, 5)
    return ans

def GrandMean(n : List[List[float]]) -> float:
    ans = 0
    for i in range((len(n))):
        for j in range(len(n[i])):
            ans += n[i][j]
          
    ans = ans / (len(n) * len(n[0]))  
    ans = round(ans, 2)
    return ans 

def mean(n: List[float]) -> float:
    ans = 0
    for item in n:
        ans += item
        
    ans = ans / (len(n))
    ans = round(ans, 2)
    return ans 

def standarddeviationsquare(n: List[float]) -> float:
    xfull = mean(n)
    ans = 0
    for items in n:
        temp = items - xfull
        temp = temp * temp
        ans += temp
        
    ans = ans / (len(n) - 1)  
    ans = round(ans, 2)   
    return ans

def sum(n: List[float]) -> float:
    ans = 0
    for items in n:
        ans += items
        
    return ans

def sq(a: float) -> float:
    return a * a

def maxoccur(n: List[float]) -> float:
    maxi = 0
    maxele = 0
    
    for i in n:
        count = 0
        for j in n:
            if i == j:
                count += 1
        
        if count > maxi:
            maxi = count
            maxele = i
            
    if maxi == 1:
        return 4999999
    
    
    return maxele

app = Flask(__name__)

@app.route('/Poisson', methods=['GET'])

def Poisson():
    x = request.args.get('x', type=float)
    lamda = request.args.get('lamda', type=float)
    if x is None or lamda is None:
        return jsonify({'error': 'Missing required parameters'}), 400
    print(x)
    print(lamda)
    equal = pow(lamda, x)
    temp = math.exp(lamda * -1)
    equal *= temp
    temp = Factorial(x)
    equal /= temp
    equal = round(equal, 5)
    less = 0
    count = 0
    while count < x:
        ans = pow(lamda, count)
        temp = math.exp(lamda * -1)
        ans *= temp
        temp = Factorial(count)
        ans /= temp
        ans = round(ans, 5)   
        count += 1
        less += ans
        
    less = round(less, 5)
    lessequal = equal + less
    lessequal = round(lessequal, 5)
    greater = 1 - lessequal
    greater = round(greater, 5)
    greaterequal = equal + greater
    greaterequal = round(greaterequal, 5)
    print(equal)
    print(less)
    print(lessequal)
    print(greater)
    print(greaterequal)
    return jsonify(
        {
            'equal': equal,
            'less': less,
            'lessequal': lessequal,
            'greater': greater,
            'greaterequal': greaterequal
        })

@app.route('/Binomial', methods=['GET'])

def Binomial():
    n = request.args.get('n', type=float)
    x = request.args.get('x', type=float)
    p = request.args.get('p', type=float)
    q = 1 - p   
    ans = Combination(n, x)
    ans *= pow(p, x)
    ans *= pow(q, n - x)
    ans = round(ans, 5)
    equal = ans
    less = 0
    count = 0
    while count < x:
        ans = Combination(n, count)
        ans *= pow(p, count)
        ans *= pow(q, n - count)
        ans = round(ans, 5)
        less += ans
        count += 1
        
    less = round(less, 5)
    lessequal = equal + less
    lessequal = round(lessequal, 5)
    greater = 1 - lessequal
    greater = round(greater, 5)
    greaterequal = equal + greater
    greaterequal = round(greaterequal, 5)
    
    return jsonify(
        {
            'equal': equal,
            'less': less,
            'lessequal': lessequal,
            'greater': greater,
            'greaterequal': greaterequal
        })

@app.route('/Multinomial', methods=['GET'])

def Multinomial():
    n = request.args.get('n', type=float)
    x = request.args.getlist('x', type=float)
    p = request.args.getlist('p', type=float)
    temp = 1
    for person in x:
        temp *= Factorial(person)
        
    print(x)
    print(p)    
    
    for probability in p:
        print(probability)
        
    for xi in x:
        print(xi)
        
    ans = Factorial(n)
    ans /= temp 
        
    for i in range(len(p)):
        ans *= pow(p[i], x[i])
        
    ans = round(ans, 5)
    return jsonify(
        {
            'ans': ans
        })

@app.route('/BayesRule', methods=['GET'])

def BayesRule():
    pa = request.args.get('pa', type=float)
    pb = request.args.get('pb', type=float)
    pab = request.args.get('pab', type=float)
    
    ans = pa * pab
    ans = ans / pb
    
    ans = round(ans, 5)
    
    return jsonify(
        {
            'ans': ans
        })

@app.route('/Anova', methods=['GET'])

def anova():
    n = request.args.getlist('n', type=float)
    sl = request.args.get('sl', type=float)
    size = int(request.args.get('size'))
    
    p = []
    l = []
    count = 1
    for item in n:
        if count != size:
            l.append(item)
            count += 1
        else:
            l.append(item)
            p.append(l)
            l = []
            count = 1
    
    print(p)
    #s2b
    
    xgm = GrandMean(p)
    kmeans = []
    for arrays in p:
        kmeans.append(mean(arrays))
        
    MSB = 0
    for means in kmeans:
        temp = (means - xgm)
        temp = temp * temp
        temp = temp * len(p[0])
        MSB += temp
        
    MSB = MSB / (len(p) - 1)

    MSB = round(MSB, 2)
    SSB = MSB * (len(p) - 1)
    SSB = round(SSB, 2)
    
    #s2w
    
    MSW = 0
    mul = len(p[0]) - 1
    for items in p:
        temp = standarddeviationsquare(items) * mul
        MSW += temp
        
    MSW = MSW / ((len(p) * len(p[0])) - len(p))
    MSW = round(MSW, 2)
    
    SSW = MSW * ((len(p[0]) - 1) * (len(p)))
    SSW = round(SSW, 2)
    
    anova = MSB / MSW
    anova = round(anova, 2)
    
    decision = scipy.stats.f.ppf(q=1-sl, dfn=(len(p) - 1), dfd=((len(p[0]) - 1) * (len(p))))
    
    decision = round(decision, 2)
    
    if anova > decision:
        hypothesis = "RejectHo"
    else:
        hypothesis = "AcceptHo"
    
    return jsonify({
        'MSB': MSB,
        'SSB': SSB,
        'dfB': (len(p) - 1),
        'MSW': MSW,
        'SSW': SSW,
        'dfW': ((len(p[0]) - 1) * (len(p))),
        'fratio': anova,
        'hypothesis': hypothesis
    })
    
@app.route('/SLR', methods=['GET'])    

def SLR():
    n = int(request.args.get('n'))
    x = request.args.getlist('x', type=float)
    y = request.args.getlist('y', type=float)
    alpha = request.args.get('alpha', type=float)
    tail_str = request.args.get('tail', default='false')
    tail = tail_str.lower() in ['true', '1', 'yes']
    
    print(tail)
    
    df = n - 2
    
    sumx = sum(x)
    sumy = sum(y)
    
    xy = []
    x2 = []
    y2 = []
    
    for i in range(len(x)):
        xy.append(x[i] * y[i])
        x2.append(sq(x[i]))
        y2.append(sq(y[i]))
        
    sumxy = sum(xy)
    sumx2 = sum(x2)
    sumy2 = sum(y2)
    
    r = 0
    r = n * sumxy
    r = r - (sumx * sumy)
    r = r / (math.sqrt(((n * sumx2) - sq(sumx)) * ((n * sumy2) - sq(sumy))))
    
    r = round(r, 3)
    
    T = r * (math.sqrt((n - 2) / (1 - sq(r))))
    T = round(T, 3)
    
    if tail == True:
        check = t.ppf(1 - alpha, df)
        print("Not hello")
    elif tail == False:
        print("hello")
        check = t.ppf(1 - alpha / 2, df)
        
    if T > check or (T < (check * -1)):
        hype = "RejectHo"
    else:
        hype = "AcceptHo"
        
    a = ((sumy * sumx2) - (sumx * sumxy))
    a = a / ((n * sumx2) - (sq(sumx)))
    
    b = ((n * sumxy) - (sumx * sumy))
    b = b / ((n * sumx2) - (sq(sumx)))
    
    a = round(a, 3)
    b = round(b, 3)
    
    exp = str(a) + "+" + str(b) + "x"
    
    return jsonify({
        'r': r,
        't': T,
        'hypothesis': hype,
        'Y': exp
    })

@app.route('/Ungrouped', methods=['GET'])

def Ungrouped():
    n = request.args.getlist('n', type=float)
    
    sl = {}
    
    for item in n:
        if item < 10:
            if "0" not in sl:
                sl["0"] = []
            sl["0"].append(item)
        else:
            temp = item % 10
            mp = item - temp
            mp = mp / 10
            key = str(mp)
            if key not in sl:
                sl[key] = []
            sl[key].append(temp)
    
    mean = mean(n)
    
    n.sort()
    
    median = statistics.median(n)
        
    val = maxoccur(n)
    
    if val == 4999999:
        mode = ""
    else:
        mode = str(val)
        
    variance = standarddeviationsquare(n)
    sd = math.sqrt(variance)
    
    sd = round(sd, 3)
    
    SoaD = ""
    
    if mean == median:
        SoaD = "Symmetric"
        
    elif mean < median:
        SoaD = "Left-Skewed"
        
    else:
        SoaD = "Right-Skewed"
        
    countone = 0
    counttwo = 0
    countthree = 0
    
    for item in n:
        if item > (mean - sd) and item < (mean + sd):
            countone += 1
            
        if item > (mean - (2 * sd)) and item < (mean + (sd * 2)):
            counttwo += 1
            
        if item > (mean - (3 * sd)) and item < (mean + (sd * 3)):
            countthree += 1
            
    data = np.array(n)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
        
    return jsonify({
        'mean': mean,
        'median': median,
        'mode': mode,
        'sd': sd,
        'Shape_of_the_Distribution': SoaD,
        'variance': variance,
        'one': ((countone / len(n)) * 100),
        'two': ((counttwo / len(n)) * 100),
        'three': ((countthree / len(n)) * 100),
        'q1': q1,
        'q3': q3,
        'stemleaf': sl
    })

@app.route('/Grouped', methods=['GET'])

def Grouped():
    first = request.args.getlist('first', type=float)
    second = request.args.getlist('second', type=float)
    frequency = request.args.getlist('freq', type=float)
    
    midpoint = []
    fm = []
    xmean = []
    xmean2 = []
    fxmean2 = []
    
    for i in range(len(first)):
        midpoint.append((first[i] + second[i]) / 2)
        
    sumf = sum(frequency)
    
    for i in range(len(frequency)):
        fm.append(frequency[i] * midpoint[i])
        
    sumfm = sum(fm)
    
    mean = sumfm / sumf
    
    mean = round(mean, 3)
    
    val = max(frequency)
    
    k = 0
    
    for j in range(len(frequency)):
        if frequency[j] == val:
            k = j
            
    mode = str(first[k]) + "-" + str(second[k])
    
    cf = [frequency[i]]
    
    for i in range(1, len(frequency)):
        cf.append(cf[i - 1] + frequency[i])
        
    med = sumf / 2
    
    for i in range(1, len(cf)):
        if med > cf[i - 1] and med < cf[i]:
            median = str(first[i]) + "-" + str(second[i])
            
    for i in range(len(midpoint)):
        xmean.append(midpoint[i] - mean)
        
    for item in xmean:
        xmean2.append(sq(item))
        
    for i in range(len(frequency)):
        fxmean2.append(frequency[i] * xmean2[i])
        
    fxmean2sum = sum(fxmean2)
    
    sd = math.sqrt((fxmean2sum / (sumf - 1)))
    
    sd = round(sd, 3)
    variance = sq(sd)
    
    variance = round(variance, 3)
        
    return jsonify({
        'mean': mean,
        'mode': mode,
        'median': median,
        'sd': sd,
        'variance': variance
    })
    
@app.route('/Hypothesis', methods=['GET'])

def Hypothesis():
    hmean = request.args.get('hmean', type=float)
    smean = request.args.get('smean', type=float)
    n = request.args.get('n', type=float)
    sl = request.args.get('sl', type=float)
    sd = request.args.get('sd', type=float)
    tail_str = request.args.get('tail', default='false')
    tail = tail_str.lower() in ['true', '1', 'yes']
    sm_str = request.args.get('samplem', default='false')
    sm = sm_str.lower() in ['true', '1', 'yes']  
    
    if n > 30:
        temp = 1 + sl
        temp = temp / 2
        zvalue = norm.ppf(temp)
        z = 0
        z = smean - hmean
        z = z / (sd / math.sqrt(n))
        if z > zvalue or z < zvalue:
            hype = "RejectHo"
        else:
            hype = "AcceptHo"
        z = round(z, 2)
        return jsonify({
            'hypothesis': hype
        })
    else:
        df = n - 1
        if tail == True:
            check = t.ppf(1 - sl, df)
        elif tail == False:
            check = t.ppf(1 - sl / 2, df)
            
        T = smean - hmean
        T = T / (sd/math.sqrt(n))
        if T > check or T < check:
            hype = "RejectHo"
        else:
            hype = "AcceptHo"
        T = round(T, 2)
        return jsonify({
            'hypothesis': hype
        })
    
if __name__ == '__main__':
    app.run(port=5000)
