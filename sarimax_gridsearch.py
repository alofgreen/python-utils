
import statsmodels.api as sm

def evaluate_models(exogDF, 
                    endogDF, 
                    p_values, 
                    d_values, 
                    q_values,
                    seasonal_gridsearch=None, 
                    seasonal_order_values=None, 
                    p_seasonal=None, 
                    q_seasonal=None, 
                    d_seasonal=None, 
                    s_seasonal=None,
                    verbose=True):
        
    """
    Grid search p, d and q values for a SARIMAX (Seasonal ARIMA with exogenious regressors)
    exogDF: exogenious regressor variables.  DataFrame
    endogDF:  target variable with date index. DataFrame
    p_values: AR parameters to search
    d_values: nonseasonal differencing parameters to search 
    q_values: lagged forecast errors to search
    seasonal_order_values: (p,d,q,s) parameters
    If necessary you can do a grid search on the seasonal_order values as well:
        p_seasonal = AR seasonal parameters
        q_seasonal = lagged forcast errors for seasonal component
        d_seasonal = seasonal differencing parameters
    
    Packages needed
        import itertools.product
        import statsmodels.api as sm (VERSION 0.8.0 OR LATER)
        import statsmodels.tsa as tsa (VERSION 0.8.0 OR LATER)
        import pandas as pd
    """
    aicList = []    
    if seasonal_gridsearch is None:
        
        orderTest = list([p,d,q])
        for val in orderTest:
            try:
                if all(isinstance(x, int) for x in val)==False:
                    raise ValueError
                    break
            except ValueError:
                print('p,d and q values must be list of integers')
        
        best_score, best_order = float("inf"), None
        allOrders_nonseasonal = list(itertools.product(p_values,d_values,q_values))
        
        for orderVals in allOrders_nonseasonal:
            non_seasonal_order_values = orderVals
            if seasonal_order_values is None:
                try:
                    mod = sm.tsa.statespace.SARIMAX(endogDF, exogDF, order=non_seasonal_order_values)
                    res = mod.fit(disp=False)
                    aic = res.aic
                    aicList.append(aic)
                    if verbose is True:
                        print(('Order={0} AIC={1}').format(non_seasonal_order_values,aic))
                    if aic<best_score:
                        best_score, best_order = aic, non_seasonal_order_values
                except:
                    continue
            else:
                try:
                    mod = sm.tsa.statespace.SARIMAX(endogDF, exogDF, order=non_seasonal_order_values, seasonal_order=seasonal_order_values)
                    res = mod.fit(disp=False)
                    aic = res.aic
                    aicList.append(aic)
                    if verbose is True:
                        print(('Order={0}, Seasonal_Order={1} AIC={2}').format(non_seasonal_order_values, seasonal_order_values, aic))
                    if aic<best_score:
                        best_score, best_order = aic, non_seasonal_order_values
                except:
                    continue
                    
    if seasonal_gridsearch is True:
        
        orderTest = list([p_values,d_values,q_values,p_seasonal,d_seasonal,q_seasonal,s_seasonal])
        for val in orderTest:
            try:
                if all(isinstance(x, int) for x in val)==False:
                    raise ValueError
                    break
            except ValueError:
                print('p,d and q values must be list of integers')
                
        best_score, best_order, best_seasonal_order = float("inf"), None, None
        
        allpdq = list(itertools.product(p_values,d_values,q_values))
        allpdqs = list(itertools.product(p_seasonal,q_seasonal,d_seasonal,s_seasonal))
        allOrders_seasonal = list(itertools.product(allpdq,allpdqs))

        for orderVals in allOrders_seasonal:
            non_seasonal_order_values = orderVals[0]
            seasonal_order_values_GS = orderVals[1]
            try:
                mod = sm.tsa.statespace.SARIMAX(endogDF, exogDF, order=non_seasonal_order_values, seasonal_order=seasonal_order_values_GS)
                res = mod.fit(disp=False)
                aic = res.aic
                aicList.append(aic)
                if verbose is True:
                    print(('Order={0}, Seasonal_order={1} AIC={2}').format(non_seasonal_order_values, seasonal_order_values_GS, aic))
                if aic<best_score:
                    best_score, best_order, best_seasonal_order = aic, non_seasonal_order_values, seasonal_order_values_GS
            except:
                continue
                
    allAICs = pd.Series(aicList)
    aicAvg = allAICs.mean()
    aicMin = allAICs.min()
    aicMax = allAICs.max()

    if seasonal_gridsearch is None and seasonal_order_values is None:
        print(('Best Model:Order={0} AIC={1}; Avg AIC={2}, Min AIC={3}, Max AIC={4}').format(best_order,best_score, aicAvg, aicMin, aicMax))
        finalMod = sm.tsa.statespace.SARIMAX(endogDF, exogDF, order=best_order)
        return [finalMod, best_order, best_score]
        
    elif seasonal_gridsearch is None and seasonal_order_values is not None:
        print(('Best Model:Order={0}, Seasonal Order={5} AIC={1}; Avg AIC={2}, Min AIC={3}, Max AIC={4}').format(best_order,best_score, aicAvg, aicMin, aicMax, seasonal_order_values))
        finalMod = sm.tsa.statespace.SARIMAX(endogDF, exogDF, order=best_order, seasonal_order=seasonal_order_values)
        return [finalMod, best_order, seasonal_order_values, best_score]
        
    elif seasonal_gridsearch is True:
        print(('Best Model:Order={0},Seasonal Order={5} AIC={1};Avg AIC={2},Min AIC={3},Max AIC={4}').format(best_order,best_score, aicAvg, aicMin, aicMax, best_seasonal_order))
        finalMod = sm.tsa.statespace.SARIMAX(endogDF, exogDF, order=best_order, seasonal_order=best_seasonal_order)
        return [finalMod, best_order, best_seasonal_order, best_score] 