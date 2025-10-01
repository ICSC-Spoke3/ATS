The file fNARfunctions.py contains useful functions to apply the NHAR methodology.

The file fRobFunctions.py contains useful functions to apply the COM and HAR methodologies.

The file DemoTESTendTOend.py presents an end to end test to see how the COM, HAR and NHAR methodologies perform anomaly detection for a simulated dataset generated according with the attached paper.

The file fDemoFunctions.py contains the functions to apply the end to end test in DemoTESTendTOend.py via the simulations of synthetic time series.
Below the two main functions:


   
mY, mOtlrs = fSimContaminatedSeries(cn,cd,prct=0.4,cnj= np.array([80, 100, 120]),outlier='addout',tim,seas))

    Description
    -----------
    Simulate the INPUT as contaminated cd x cn matrix with cd time series along cn observations.
    You can choose the percentage of contaminated series, the tipology of outliers and their
    location.


    Parameters
    ----------
    cn :  number of observations (minimum value 121)
    cd : number of time series   (minimum value 10)
    prct : percentage of contaminated series. The default is 40%.
    cnj : Location in time of the outliers. The default is np.array([80, 100, 120]).
    outlier : outliers typology (for instance, 'addout' and 'levshif') The default is 'addout'.
    tim  : outlier effect in terms of the weight (tim) multiplied for the time series variance
    seas: seasonality in case of a seasonal outlier

    Returns
    -------
    mY : Matrix of contaminated time series.
    mOtlrs : Matrix of real outlier location 



df_mOuCOMall, df_mOuREG, df_mOuNAR   =fDemo(mY,mOtlrs,strctmdl=True,fq= 2 * np.pi / 30, fw=2 * np.pi / 7,TREND=2, COM=True, HAR=True, NHAR=True)


    Description
    -----------
    Obtain and PRINT the OUPUT in terms of percentage of the total outliers 
    detected by the COM, HAR an NHAR methodologies, along with the number of false positive.
    NOTE: The function 'fDemo' contains the individual functions to run the COM, HAR an NHAR methodologies singularly in case you want to split them, 
    s.t. 'fCOM(mRes,mDRes)' run the COM method, 'fARcorsi(mRes, mDRes)' run the HAR method and 'fNHAR(mRes , mDRes,          epochs=50)'
    run the NHAR method. The input are the matrix nxd of d time series with n obs ('mRes') and its first difference ('mDRes'). 
    The first output for all functions is the array of outliers locations.

    Parameters
    -------
    mY : Matrix of contaminated time series.
    mOtlrs : Matrix of real outlier location 
    strctmdl: if it is True the analysis is performed on the robust residuals 
    obtained by removing a trend plus cycle from the data.
    
    TREND and CYCLE parameters if strctmdl is TRUE
    
    fw = first set of frequencies in (0,pi) (by default is the weekly frequency)
    fq = further set of frequencies in (0,pi) (by default is the monthly frequency)
    TREND = trend component. Values allowed 0 (only intercept), 1 (deterministic trend),
    2 (quadratic trend).
       
    Methods   
    
    COM = if True the COM method is applied
    HAR = if True the HAR method is applied
    NHAR = if True the NHAR method is applied       
        
    Returns
    -------
    df_mOuCOMall : Location of the outliers detected by the COM estimator.
    df_mOuREG : Location of the outliers detected by the HAR method.
    df_mOuNAR : Location of the outliers detected by the NHAR method.


