# -*- coding: utf-8 -*-

import numpy as np
import jcamp as j
import collections
import os
import pandas as pd
import openpyxl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scp

# Written by Bram Kappé
# 04/2021, python 3.9
# b.t.kappe@uu.nl



# TO DO
# Add peak fitting
# figure out if spectrum substraction is okay now (cooling down goes slower than planned?)
# add normalization based on weight for intensity of spectra


# ADJUST THESE PARAMETERS

# Do you want to normalise/compensate for the sample weight?
NormaliseToWeight = True
# Weight of Sample in MG
WeightOfSample = 18.5
# Nickel loading as a fraction (e.g. 11.8 % is 0.0118)
NickelLoading = 0.118

# Baseline correction
# Do you want to subtract spectra (based on spectra taken after reduction step)
DoSpectrumSubstraction = True

# Do you want to do a baseline correciton
DoBaseLineCorrectionJim = False

DoBaseLineCorrectionRobin = False

DoBaseLineCorrectionRobinCheck = False
NumbersToCheckRobin = [20,50]

DoBaseLineCorrectionJimCheck = False
NumbersToCheckJim = [20,50]

# What to plot? This gives the 3d line plots, use for checking the inbetween steps. I recommend setting this to true if things look weird.
# This overrides all settings with 'old' in them
PlotToCheck = True

# Do you want to plot all raw data
ShowRawDataOld = True

# Do you want to plot the relevant spectra (after starting catalysis)
ShowCutDataOld = True

# Do you want to plot the spectra that will be substracted (to check for noise)
ShowSpectraToBeSubstractedOld = True

# Do you want to plot the spectra after substraction
ShowSubstractedDataOld = True

# Do you want to plot the spectra after baseline correction (final)
ShowCorrectedDataOld = False

# What to plot?
ShowCutData = False

ShowCorrectedDataJim = True

ShowCorrectedDataRobin = True

ShowSubstractedData = True

# What wavenumbers and absorption to plot?
WaveNumberUpperLimit = 2250
WaveNumberLowerLimit = 1400
AbsorptionLowerLimit = 0
AbsorptionUpperLimit = 0.25



# Exporting data as XlSX
# Do you want to export all data as .xlsx file?
PrintRawData = False

# Do you want to export the relevant spectra without baseline correction?
PrintCutData = False

# Do you want to export the substracted spectra?
PrintSubstractedData = True

# Do you want to export the baseline corrected spectra? (final spectra)
PrintCorrectedData = False

#Autopick Starting Spectrum
# Do you want the script to pick the start of catalysis?
AutoFindStartCatalysis = True

# Do you want a plot of the Co2 signal to check if it seems right?
ShowStartFindPlot = True

# If false, please enter starting spectrum here
ManualStartSpectrum = 0

# If true, here are the values for the Co2 signal and the threshold for intensity
Co2PeakPositon = 2362
Co2IntensityTreshold = 1.7
CompensateForReductionSignal = False

#Info for how long the catalysis goes on
#time spent ramping up before catalysis
RampUpTime = 40

#time of actual catalysis
CatalyseTijd = 90

# Time spent at stable temperature before ramping up for catalysis withoutCO2
TimeAtStableTempAfterRedWithoutCo2 = 0

# Time spent at stable temperature before ramping up for catalysis with CO2
TimeAtStableTempAfterRedWithCo2 = 10


# Stable temperture which you've spent that time at
StableTemp = 200

# Speed of ramping down after reduction in degrees celcius per minute
RampDownAfterReductionSpeed = 10

# Speed of ramping up to catalysis in degrees celcius per minute
RampUpSpeed = 5

# Temperature during catalysis
CatalysTemperature = 400

# Does your baseline look weird? Set this to true and adjust parameters below
BaseLineIsWeird = False
CutOfBaseLineLowerTemperatureLimit = 200
CutOfBaseLineUpperTemperatureLimit = 350

# Just takes starting spectrum
JustTakeStartSpectrum = False

# This says how many spectrum to show after starting spectrum
RampEnCatalyse = RampUpTime + CatalyseTijd

# END OF PARAMETERS YOU NEED TO ADJUST
justtakestartspectrumwarning = True
baselineisweirdwarning = True
FoundName = False

path = os.path.dirname(os.path.abspath(__file__))
filelist = []

for root, dirs, files in os.walk(path, topdown=True):
    for name in files:
        if '.dx' in name:
            if not FoundName:
                NameOfSample = name
                FoundName = True
            filelist.append(os.path.join(root, name))

# print(NameOfSample)
NameOfSampleKort = NameOfSample.replace('0000.dx','')
# print(NameOfSampleKort)
firsttime = True

for files in filelist:
    if firsttime:
        FirstSpectrum = j.JCAMP_reader(files)
        df = pd.DataFrame({'Wavenumber': FirstSpectrum['x']})
        print('Naam van eerste file: ' + files)
        firsttime = False
    CurrentSpectrum = j.JCAMP_reader(files)
    FilesKort = files.replace('.dx','')
    CurrentNumber = FilesKort[-4:]
    if CurrentNumber[-2:] == '00':
        print(CurrentNumber)
    if '0' in CurrentNumber:
        df[CurrentNumber] = CurrentSpectrum['y'].tolist()

if PrintRawData:
    df.to_excel(NameOfSampleKort + 'rawdata.xlsx')

print('raw data:')
print(df)

Wavenumbers = df['Wavenumber']
Wavenumbers_lijst = Wavenumbers.tolist()
AantalSpectra = len(df.columns)
print('Aantal spectra:')
print(AantalSpectra)

AantalSpectraArray = np.arange(1,AantalSpectra)

def three_d_plot(dfinput, title, x_axis, z_axis_length, set_limits, x_axis_left_lim, x_axis_right_lim, y_axis_low_lim, y_axis_up_lim, transformed_df):
    ZArray = np.arange(1,z_axis_length)
    Plot = plt.figure()
    PlotAx = Plot.add_subplot(111, projection='3d')
    if not transformed_df:
        z_number = z_axis_length - 1
        for i in ZArray:
            PlotAx.plot(x_axis,dfinput.iloc[:, z_number], z_number)
            z_number -= 1
    if transformed_df:
        for i in ZArray:
            PlotAx.plot(x_axis,dfinput.iloc[i, :], z_axis_length-i)
    PlotAx.view_init(80,-90)
    PlotAx.title.set_text(title)
    if set_limits:
        PlotAx.set(xlim=(x_axis_left_lim,x_axis_right_lim), ylim=(y_axis_low_lim,y_axis_up_lim))
    # print('New plot fuction used for plot: '+ title)

def listfromspectrumnumber(spectrumnumber):
    Temp_Df = df.iloc[:,spectrumnumber]
    return Temp_Df.tolist()

def nparrayfromspectrumnumber(spectrumnumber):
    return df.iloc[:,spectrumnumber]

def findindexbywavenumber(wavenumber):
    # print('Wavenumber aangeroepen: ')
    # print(wavenumber)
    foundwavenumberintuple = np.where((Wavenumbers > (wavenumber - 2)) & (Wavenumbers < (wavenumber + 2)))
    #  print(foundwavenumberintuple)
    foundwavenumberinarray = foundwavenumberintuple[0]
    # print(foundwavenumberinarray)
    foundwavenumber = foundwavenumberinarray[0]
    # print('FindIndexByWavenumber finds: ')
    # print(foundwavenumber)
    return foundwavenumber

def findindexbywavenumber_flexible(wavenumber, dataframe):
    # print('Wavenumber aangeroepen: ')
    # print(wavenumber)
    foundwavenumberintuple = np.where((dataframe > (wavenumber - 2)) & (dataframe < (wavenumber + 2)))
    #  print(foundwavenumberintuple)
    foundwavenumberinarray = foundwavenumberintuple[0]
    # print(foundwavenumberinarray)
    foundwavenumber = foundwavenumberinarray[0]
    #print('FindIndexByWavenumber Flexible finds: ')
    #print(foundwavenumber)
    return foundwavenumber

def intensitylistfromwavenumber_nottransposed(DataFrame,WavenumberFI,ShowPlot):
    DfWavenumberFI = DataFrame.loc[findindexbywavenumber(WavenumberFI)]
    DfWavenumberFI_temp = DfWavenumberFI.iloc[1:]
    DfWavenumberFIList = DfWavenumberFI_temp.tolist()
    if ShowPlot:
        DfWavenumberFIPlot = plt.figure()
        DfWavenumberFIPlotAx = DfWavenumberFIPlot.add_subplot()
        DfWavenumberFIPlotAx.plot(AantalSpectraArray,DfWavenumberFIList)
        DfWavenumberFIPlotAx.title.set_text('Plot of intensity of peak at ' + str(WavenumberFI))
    return DfWavenumberFIList

def intensitylistfromwavenumber(DataFrame,WavenumberFI,ShowPlot, Export):
    # AantalSpectraArrayFI = np.arange(1,len(DataFrame.rows))
    RoundWaveNumberFI = find_index_of_nearest_round_wavenumber(WavenumberFI)
    # print('Round wavenumber: '+ str(RoundWaveNumberFI))
    DfWavenumberFI = DataFrame.iloc[:,RoundWaveNumberFI]
    #  print(DfWavenumberFI)
    DfWavenumberFIList = DfWavenumberFI.tolist()
    # print(DfWavenumberFIList)
    print('Plotting intensity of signal at ' + str(RoundWaveNumberFI))
    if ShowPlot:
        DfWavenumberFIPlot = plt.figure()
        DfWavenumberFIPlotAx = DfWavenumberFIPlot.add_subplot()
        DfWavenumberFIPlotAx.plot(DfWavenumberFIList)
        DfWavenumberFIPlotAx.title.set_text('Plot of intensity of peak at ' + str(RoundWaveNumberFI))
        DfWavenumberFIPlotAx.set_xlabel('Spectrum Number', fontsize=10)
        DfWavenumberFIPlotAx.set_ylabel('Intensity', fontsize=10)
    if Export:
        TempDfForExport_Intensity = pd.DataFrame()
        TempDfForExport_Intensity['Intensity list'] = DfWavenumberFIList
        print(TempDfForExport_Intensity)
        TempDfForExport_Intensity.to_excel(NameOfSampleKort + 'Plot of intensity of peak at ' + str(RoundWaveNumberFI) + '.xlsx')
    return DfWavenumberFIList

def findstartofsignal(ListToCheck,MinimumIntensity,TimerTime,GiveSecondTime):
    if GiveSecondTime == True:
        BeforeFirstTime = True
        AfterFirstTime = False
    if GiveSecondTime == False:
        BeforeFirstTime = False
        AfterFirstTime = True
    Timer = TimerTime
    for i in ListToCheck:
        if BeforeFirstTime == True:
            if i >= MinimumIntensity:
                BeforeFirstTime = False
        if  BeforeFirstTime == False:
            if AfterFirstTime == False:
                if i < MinimumIntensity:
                    Timer -= 1
                if Timer <= 0:
                    AfterFirstTime = True
                    print('After Start')
            if AfterFirstTime == True:
                    if i >= MinimumIntensity:
                        return ListToCheck.index(i)

if AutoFindStartCatalysis:
    ListofCo2peak = intensitylistfromwavenumber_nottransposed(df,Co2PeakPositon,ShowStartFindPlot)
    # plt.show()
    StartSpectrum = findstartofsignal(ListofCo2peak,Co2IntensityTreshold,10,CompensateForReductionSignal)
    if StartSpectrum == None:
        print('Did not find starting spectrum!')
        plt.show()
    print('Automatically found starting spectrum: ')
    print(StartSpectrum)

def findtemperatureofspectra(SpectrumNumber):
    RelativeNumber = SpectrumNumber - StartSpectrum - TimeAtStableTempAfterRedWithCo2

    if RelativeNumber < (-TimeAtStableTempAfterRedWithCo2-1):
        print('error: spectrum has to be after start')
        return 0
    if RelativeNumber <= 0:
        return StableTemp
    if RelativeNumber >= RampUpTime:
        return CatalysTemperature
    if RelativeNumber < RampUpTime:
        return round((RelativeNumber*RampUpSpeed) + StableTemp)
    print('Error - Cant find temperature of spectra!')

def baseline_correction_jim(raman_spectra,niter):

    assert(isinstance(raman_spectra, pd.DataFrame)), 'Input must be pandas DataFrame'

    spectrum_points = len(raman_spectra.columns)
    raman_spectra_transformed = np.log(np.log(np.sqrt(raman_spectra +1)+1)+1)

    working_spectra = np.zeros(raman_spectra.shape)

    for pp in np.arange(1,niter+1):
        r1 = raman_spectra_transformed.iloc[:,pp:spectrum_points-pp]
        r2 = (np.roll(raman_spectra_transformed,-pp,axis=1)[:,pp:spectrum_points-pp] + np.roll(raman_spectra_transformed,pp,axis=1)[:,pp:spectrum_points-pp])/2
        working_spectra = np.minimum(r1,r2)
        raman_spectra_transformed.iloc[:,pp:spectrum_points-pp] = working_spectra

    baseline = (np.exp(np.exp(raman_spectra_transformed)-1)-1)**2 -1
    return baseline

def baseline_correction_robin(y, lam, p, niter=10):
  L = len(y)
  D = scp.sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = scp.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scp.sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def findspectrafortemperature(Temperature):
    RelativeTemperature = Temperature - StableTemp
    if JustTakeStartSpectrum:
        if justtakestartspectrumwarning:
            print('Taking starting spectrum as baseline correction!')
        return StartSpectrum
    if Temperature == StableTemp:
        return StartSpectrum
    if Temperature < StableTemp:
        print('Error: Temperature called (' + str(Temperature) + ' is below stable temperature')
        return 0
    if BaseLineIsWeird:
        if baselineisweirdwarning:
            print('Correcting baseline!')
            print('Lower limit, upper limit is:')
            print(CutOfBaseLineLowerTemperatureLimit)
            print(CutOfBaseLineUpperTemperatureLimit)
        if Temperature < CutOfBaseLineUpperTemperatureLimit:
           if Temperature > CutOfBaseLineLowerTemperatureLimit:
                if Temperature > StableTemp:
                    EndOfRampDown = StartSpectrum - TimeAtStableTempAfterRedWithoutCo2
                    return EndOfRampDown - round(RelativeTemperature/RampDownAfterReductionSpeed)
        if Temperature < CutOfBaseLineUpperTemperatureLimit:
            if Temperature <= CutOfBaseLineLowerTemperatureLimit:
                return StartSpectrum
        if Temperature >= CutOfBaseLineUpperTemperatureLimit:
            EndOfRampDown = StartSpectrum - TimeAtStableTempAfterRedWithoutCo2
            return EndOfRampDown - round(CutOfBaseLineUpperTemperatureLimit / RampDownAfterReductionSpeed)
    if Temperature > StableTemp:
        EndOfRampDown = StartSpectrum - TimeAtStableTempAfterRedWithoutCo2
        return EndOfRampDown - round(RelativeTemperature / RampDownAfterReductionSpeed)

def spectrum_from_df(DataFrame, SpectrumNumber):
    DataFrameLengte = len(DataFrame.index)
    # print('Lengte van dataframe: ' + str(DataFrameLengte))
    #  print(DataFrame)
    # print('Aangeroepen spectrum: '+ str(SpectrumNumber))
    SpectrumIndex = DataFrameLengte-SpectrumNumber
    # print('Deze staat op plek: '+ str(SpectrumIndex))
    return DataFrame.iloc[SpectrumIndex,:]

def linetraces_by_spectrum_numbers(DataFrame, SpectrumNumbers, TitleAddition, Export):
    LineTracePlot = plt.figure()
    LineTracePlotAx = LineTracePlot.add_subplot()
    if Export: LineTracesDF = pd.DataFrame()
    for i in SpectrumNumbers:
        LineTracePlotAx.plot(spectrum_from_df(DataFrame,i), label = str(i))
        if Export: LineTracesDF[str(i)] = spectrum_from_df(DataFrame,i)
    LineTracePlotAx.title.set_text('Linetraces of spectras: ' + str(SpectrumNumbers) + ' ' + TitleAddition)
    LineTracePlotAx.set(xlim=(WaveNumberUpperLimit,WaveNumberLowerLimit))
    LineTracePlotAx.set_xlabel('Wavenumber (cm-1)', fontsize=10)
    LineTracePlotAx.set_ylabel('Intensity', fontsize=10)
    LineTracePlotAx.legend()
    LineTracePlot.savefig(NameOfSampleKort + 'linetraces for spectra' + str(SpectrumNumbers) + ' ' + TitleAddition + '.png')
    print('Plotting linetraces for spectra: ' + str(SpectrumNumbers) + ' ' + TitleAddition)
    if Export: LineTracesDF.to_excel(NameOfSampleKort + 'linetraces.xlsx')

if AutoFindStartCatalysis == False:
    StartSpectrum = ManualStartSpectrum
    print('Manually set starting spectrum to: ')
    print(StartSpectrum)

# cutting data to relevent wavenumbers and spectra
KolommenDieWeHouden = np.arange(StartSpectrum, StartSpectrum + RampEnCatalyse)
RijenDieWeHouden = np.arange(findindexbywavenumber(WaveNumberUpperLimit), findindexbywavenumber(WaveNumberLowerLimit))
RelevanteSpectra = df.iloc[RijenDieWeHouden,KolommenDieWeHouden]
WavenumberDieWehouden = Wavenumbers_lijst[findindexbywavenumber(WaveNumberUpperLimit):findindexbywavenumber(WaveNumberLowerLimit)]

def subtractlist (ListA, ListB):
    NewList = []
    ListLength = len(ListA)
    ListCounter = np.arange(0,ListLength)
    for i in ListCounter:
        NewList.append(ListA[i] - ListB[i])
    return NewList

if DoSpectrumSubstraction:
    DFCorrected = pd.DataFrame({'Wavenumber': df['Wavenumber']})
    TemperaturenInBaseline = []
    DFBaselines = pd.DataFrame({'Wavenumber': df['Wavenumber']})
    BaselineCounter = 0
    for i in KolommenDieWeHouden:
        CurrentTemperature = findtemperatureofspectra(i)
        # print('current temperature is: ' + str(CurrentTemperature))
        RelevantSpectrumForBaseline = findspectrafortemperature(CurrentTemperature)
        # print('The relevant spectrum for this temperature is: ' + str(RelevantSpectrumForBaseline))
        baselineisweirdwarning = False
        justtakestartspectrumwarning = False
        BaseLineArray = nparrayfromspectrumnumber(RelevantSpectrumForBaseline)
        DataArray = nparrayfromspectrumnumber(i)
        DFCorrected[i]= np.subtract(DataArray,BaseLineArray)
        if ShowSpectraToBeSubstractedOld:
            if not CurrentTemperature in TemperaturenInBaseline:
                BaselineCounter += 1
                TemperaturenInBaseline.append(CurrentTemperature)
                DFBaselines[BaselineCounter] = BaseLineArray
            DFBaselineCut =DFBaselines.iloc[RijenDieWeHouden, :]
    DFCorrectedAndCut = DFCorrected.iloc[RijenDieWeHouden,:]

if PrintCutData:
    RelevanteSpectra.to_excel(NameOfSampleKort + 'cutdata.xlsx')

# print('Data before wavenumbers rounding')
# print(DFCorrectedAndCut)
# Transforms dataframe for heatmap
WavenumbersBeforeRounding = DFCorrectedAndCut.iloc[:, 0].tolist()
# print('Wavenumbers before rounding')
# print(WavenumbersBeforeRounding)
RoundedWavenumbers = []
for i in WavenumbersBeforeRounding:
    RoundedWavenumbers.append(int(round(i)))

# print('Wavenumbers after rounding')
# print(RoundedWavenumbers)
# DFCorrectedAndCut['Wavenumber'] = int(round(DFCorrectedAndCut['Wavenumber']))
DFCorrectedAndCut['Wavenumber'] = RoundedWavenumbers
DataBeforeTransform = DFCorrectedAndCut.set_index('Wavenumber')
SubstractedDFWithName = DataBeforeTransform.transpose()
SubstractedDFWithWrongIndex = SubstractedDFWithName.rename_axis(None, axis=1)
NewIndexForDF = np.arange(1, len(SubstractedDFWithWrongIndex.index) + 1)
SubstractedDFBeforeFlip = SubstractedDFWithWrongIndex.set_index(NewIndexForDF)
SubstractedDF = SubstractedDFBeforeFlip.iloc[::-1]
ListOfRoundedWavenumbers = list(SubstractedDF)
# SubstractedDF = SubstractedDFWithDublicateColumns.loc[:,~SubstractedDFWithDublicateColumns.columns.duplicated()]

# print('Data after wavenumbers rounding')
# print(DFCorrectedAndCut)

# print('Data before tranform')
# print(DataBeforeTransform)

# print('Substracted dataframe With Old Index')
# print(SubstractedDFWithWrongIndex)

# print('Substracted dataframe New index and Flipped')
# print(SubstractedDF)

def find_nearest_round_wavenumber(WaveNumber):
    WaveNumberIndex = min(range(len(ListOfRoundedWavenumbers)), key=lambda i: abs(ListOfRoundedWavenumbers[i]-WaveNumber))
    return ListOfRoundedWavenumbers[WaveNumberIndex]

def find_index_of_nearest_round_wavenumber(WaveNumber):
    return min(range(len(ListOfRoundedWavenumbers)), key=lambda i: abs(ListOfRoundedWavenumbers[i]-WaveNumber))

def find_maximum_in_area(Dataframe, UpperWaveNumber, LowerWaveNumber, StartingSpectrum, FinalSpectrum, PlotOrNot, Export):
    WavenumbersAtMaximum = []
    Max_ListofWavenumbers = list(Dataframe)
    SpectrumArray = np.arange(StartingSpectrum,FinalSpectrum)
    UpperWaveNumberIndex = min(range(len(Max_ListofWavenumbers)), key=lambda i: abs(Max_ListofWavenumbers[i]-UpperWaveNumber))
    UpperWaveNumberActual = Max_ListofWavenumbers[UpperWaveNumberIndex]
    LowerWaveNumberIndex = min(range(len(Max_ListofWavenumbers)), key=lambda i: abs(Max_ListofWavenumbers[i]-LowerWaveNumber))
    LowerWaveNumberActual = Max_ListofWavenumbers[LowerWaveNumberIndex]
    # print('Current dataframe: ')
    # print(Dataframe)
    print('Finding maximum from wavenumbers ' + str(UpperWaveNumberActual) + ' to ' + str(LowerWaveNumberActual) + ' with corresponding indexes: ' + str(UpperWaveNumberIndex) + ' and ' + str(LowerWaveNumberIndex))
    for i in SpectrumArray:
        SpectrumForMax = spectrum_from_df(Dataframe, i)
        # print('SpectrumForMax for spectrum: ' + str(i))
        # print(SpectrumForMax)
        # print(type(SpectrumForMax))
        CutSpectrumForMax = SpectrumForMax.truncate(before = str(LowerWaveNumberActual), after = str(UpperWaveNumberActual)) # , axis = "columns"
        # print('CutSpectrumForMax')
        # print(CutSpectrumForMax)
        MaxIndexFromSpectrum = CutSpectrumForMax.idxmax()
        # print('Max Index: ' + str(MaxIndexFromSpectrum))
        # MaxWaveNumber = Dataframe.index[MaxIndexFromSpectrum]
        WavenumbersAtMaximum.append(MaxIndexFromSpectrum)
    # print(WavenumbersAtMaximum)
    # print(SpectrumArray)
    if Export:
        TempDfForExport_Max = pd.DataFrame()
        TempDfForExport_Max['Spectrum Number'] = SpectrumArray
        TempDfForExport_Max['Wavenumber of Maximum'] = WavenumbersAtMaximum
        print(TempDfForExport_Max)
        TempDfForExport_Max.to_excel(NameOfSampleKort + 'Location of maximum between wavenumbers ' + str(UpperWaveNumberActual) + ' and ' + str(LowerWaveNumberActual) + ' for spectra ' + str(StartingSpectrum) + ' to ' + str(FinalSpectrum) + '.xlsx')
    if PlotOrNot:
        MaximumPlot = plt.figure()
        MaximumPlotAx =  MaximumPlot.add_subplot()
        MaximumPlotAx.plot(SpectrumArray, WavenumbersAtMaximum)
        MaximumPlotAx.title.set_text('Location of maximum between wavenumbers: ' + str(UpperWaveNumberActual) + ' and ' + str(LowerWaveNumberActual))
        MaximumPlotAx.set_xlabel('Spectrum Number', fontsize = 10)
        MaximumPlotAx.set_ylabel('Wavenumber (cm-1)', fontsize = 10)
        MaximumPlotAx.set(xlim=(StartingSpectrum,FinalSpectrum),ylim=(LowerWaveNumber,UpperWaveNumber))
        MaximumPlot.savefig('Location of maximum between wavenumbers ' + str(UpperWaveNumberActual) + ' and ' + str(LowerWaveNumberActual) + '.png')
    return WavenumbersAtMaximum

if NormaliseToWeight:
    print('Normalizing weight with sample weight of ' + str(WeightOfSample) + ' mg!')
    print(SubstractedDF)
    SubstractedDF = SubstractedDF.div(WeightOfSample/20)
    print(SubstractedDF)

if DoBaseLineCorrectionJim:
    CorrectedDF = SubstractedDF -  baseline_correction_jim(SubstractedDF,20)
    print('Corrected dataframe via Jim Method:')
    print(CorrectedDF)
    # print('BaselineCorrectedDF')
    #  print(CorrectedDF)

if DoBaseLineCorrectionRobin:
    BaseRobin_lengte = len(SubstractedDF.index)
    # print(BaseRobin_lengte)
    BaseRobin_Array = np.arange(0,BaseRobin_lengte)
    # print(BaseRobin_Array)
    # BaseRobin_Array = [20,50]
    # BaseRobin_list = []
    RobinCorrectedDf = pd.DataFrame(index=SubstractedDF.index,columns=RoundedWavenumbers)
    # print(RobinCorrectedDf)
    for i in BaseRobin_Array:
        BaseRobin_spectrum = SubstractedDF.iloc[i,:]
        # print(BaseRobin_spectrum)
        BaseRobin_baseline = pd.Series(data=baseline_correction_robin(BaseRobin_spectrum,1000,0.001),index=RoundedWavenumbers,dtype='float16')
        # (BaseRobin_baseline)
        BaseRobin_result = BaseRobin_spectrum-BaseRobin_baseline
        RobinCorrectedDf.iloc[i,:] = BaseRobin_result
        # print(BaseRobin_result)

        # RobinCorrectedDf.to_excel('Corrected data via robin method.xlsx')
    print('Corrected dataframe via Robin Method:')
    print(RobinCorrectedDf)
    print(BaseRobin_baseline)
    RobinCorrectedDf = RobinCorrectedDf.astype('float16')

if DoBaseLineCorrectionRobinCheck:
    for i in NumbersToCheckRobin:
        baseline_test_spectrum = spectrum_from_df(SubstractedDF,i)
        baseline_test_baseline = baseline_correction_robin(baseline_test_spectrum,1000,0.001)
        baseline_test_result = baseline_test_spectrum - baseline_test_baseline
        BaselineTestPlot = plt.figure()
        BaselineTestPlotAx = BaselineTestPlot.add_subplot()
        BaselineTestPlotAx.plot(baseline_test_spectrum, label='Original spectrum', color = 'k')
        BaselineTestPlotAx.plot(RoundedWavenumbers, baseline_test_baseline, label='Baseline', color = 'r')
        BaselineTestPlotAx.plot(baseline_test_result, label='Result', color='g')
        BaselineTestPlotAx.title.set_text('Baseline test Robin method for spectrum: ' + str(i))
        BaselineTestPlotAx.set(xlim=(WaveNumberUpperLimit, WaveNumberLowerLimit))
        BaselineTestPlotAx.set_xlabel('Wavenumber (cm-1)', fontsize=10)
        BaselineTestPlotAx.set_ylabel('Intensity', fontsize=10)
        BaselineTestPlotAx.legend()
        print('Plotting baseline test (Robin method, least squares) for spectrum: ' + str(i))

if DoBaseLineCorrectionJimCheck:
    for i in NumbersToCheckJim:
        baseline_test_spectrum = spectrum_from_df(SubstractedDF, i)
        baseline_test_result = spectrum_from_df(CorrectedDF, i)
        baseline_test_baseline = baseline_test_spectrum-baseline_test_result
        BaselineTestPlot = plt.figure()
        BaselineTestPlotAx = BaselineTestPlot.add_subplot()
        BaselineTestPlotAx.plot(baseline_test_spectrum, label='Original spectrum', color='k')
        BaselineTestPlotAx.plot(RoundedWavenumbers, baseline_test_baseline, label='Baseline', color='r')
        BaselineTestPlotAx.plot(baseline_test_result, label='Result', color='g')
        BaselineTestPlotAx.title.set_text('Baseline test Jim method for spectrum: ' + str(i))
        BaselineTestPlotAx.set(xlim=(WaveNumberUpperLimit, WaveNumberLowerLimit))
        BaselineTestPlotAx.set_xlabel('Wavenumber (cm-1)', fontsize=10)
        BaselineTestPlotAx.set_ylabel('Intensity', fontsize=10)
        BaselineTestPlotAx.legend()
        print('Plotting baseline test (Jim method) for spectrum: ' + str(i))
# print(BackUpDF)
# print(Wavenumbers)
# print(spectrum_from_df(SubstractedDF, 2))
# print(spectrum_from_df(CorrectedDF, 4))
# linetraces_by_spectrum_numbers(SubstractedDF, [5,10, 25, 50, 80, 110], 'of substracted data', True)
# print(spectrum_from_df(SubstractedDF,80).to_list())
# linetraces_by_spectrum_numbers(CorrectedDF, [10,25,50,80,90], 'of Jim corrected data')
# linetraces_by_spectrum_numbers(RobinCorrectedDf, [10,25,50,80,90], 'of Robin corrected data', True)
# print('CorrectedDF:')
# print(CorrectedDF)
# print('Columns of CorrectedDF')
# print(CorrectedDF.keys())
# print(list(CorrectedDF))
# intensitylistfromwavenumber(CorrectedDF, 2040, True, True)
# find_maximum_in_area(SubstractedDF,2100,1950,2,130,True, True)

if PrintSubstractedData: SubstractedDF.to_excel(NameOfSampleKort + 'Substracted Data.xlsx')

if PrintCorrectedData: CorrectedDF.to_excel(NameOfSampleKort + 'Corrected Data.xlsx')

if PlotToCheck:
    if ShowRawDataOld:
        three_d_plot(df, 'Raw Data', Wavenumbers_lijst, AantalSpectra, False, WaveNumberUpperLimit, WaveNumberLowerLimit, AbsorptionLowerLimit, 0.2, False)

    if ShowSpectraToBeSubstractedOld:
        three_d_plot(DFBaselineCut, 'Spectra used for substraction', WavenumberDieWehouden, BaselineCounter, True, WaveNumberUpperLimit, WaveNumberLowerLimit, AbsorptionLowerLimit, 3, False)

    if ShowCutDataOld:
        three_d_plot(RelevanteSpectra, 'Cut Data', WavenumberDieWehouden, RampEnCatalyse, True, WaveNumberUpperLimit, WaveNumberLowerLimit, AbsorptionLowerLimit, 3, False)

    if ShowSubstractedDataOld:
        three_d_plot(SubstractedDF, 'Substracted Spectra', WavenumberDieWehouden, RampEnCatalyse, True, WaveNumberUpperLimit, WaveNumberLowerLimit, AbsorptionLowerLimit, 0.4, True)

    if ShowCorrectedDataOld:
       three_d_plot(CorrectedDF, 'Baseline corrected Data', WavenumberDieWehouden, RampEnCatalyse, True, WaveNumberUpperLimit, WaveNumberLowerLimit, AbsorptionLowerLimit, 0.2, True)

if ShowCutData:
    CutDataHeatMap = sns.heatmap(RelevanteSpectra)

if ShowSubstractedData:
    SubstractedHeatPlot = plt.subplots(figsize=(20, 5), ncols=1, nrows=1, squeeze=False)
    plt.title('Heatmap of Substracted Spectra')
    SubstractedHeatMap = sns.heatmap(SubstractedDF, xticklabels=15, yticklabels=10, vmin=AbsorptionLowerLimit, vmax=AbsorptionUpperLimit)
    #vmin = 0, vmax = 0.15
    plt.xlabel('Wavenumbers')
    plt.ylabel('Spectrum Number')
    plt.savefig(NameOfSampleKort + 'SubstractedHeatmap.png')
    # SubstractedHeatMap.set_ylabels('Spectrum Number')
    # SubstractedHeatMap = sns.heatmap(SubstractedDF, xticklabels=40, yticklabels="auto", ax=ax[0, 0])
if ShowCorrectedDataJim:
    CorrectedJimHeatPlot = plt.subplots(figsize=(20, 5), ncols=1, nrows=1, squeeze=False)
    plt.title('Heatmap of Corrected Spectra - Jim Method')
    CorrectedJimHeatMap = sns.heatmap(CorrectedDF, xticklabels=40, yticklabels="auto")
    plt.xlabel('Wavenumber')
    plt.ylabel('Spectrum')
    plt.savefig(NameOfSampleKort + 'JimCorrectedHeatmap.png')
    # fig2.title.set_text('Data after baseline correction')
   # vmin=AbsorptionLowerLimit, vmax=0.2
    # sns.heatmap(CorrectedDF, xticklabels=10, vmax=1000, yticklabels=40, ax=ax[0, 0], cbar_kws={'label': z}, cmap='viridis');

    # CorrectedDataHeatMap = sns.heatmap(CorrectedDF)

if ShowCorrectedDataRobin:
    CorrectedRobinHeatPlot = plt.subplots(figsize=(20, 5), ncols=1, nrows=1, squeeze=False)
    plt.title('Heatmap of Corrected Spectra - Robin Method')
    CorrectedRobinHeatMap = sns.heatmap(RobinCorrectedDf, xticklabels="auto", yticklabels="auto")
    plt.xlabel('Wavenumber')
    plt.ylabel('Spectrum')
    plt.savefig(NameOfSampleKort + 'RobinCorrectedHeatMap.png')
plt.show()







