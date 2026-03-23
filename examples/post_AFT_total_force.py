import flow360
import math, json, pandas
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import shutil

import json

from flow360.cloud.http_util import api_key_auth, http
from flow360.log import log, set_logging_level
from flow360.user_config import UserConfig

import flow360 as fl

import glob

import os
import pylatex
import re
import shutil

SOLVER_VERSION = os.environ.get("solverVersion", "")
SOLVER_VERSION_REF = os.environ.get("solverVersionRef", "")
#RESULTS_PATH = os.environ.get("./esultsPath", None)

#RESULTS_PATH = os.environ.get("./", None)
#print("RESULTS_PATH=", RESULTS_PATH)
RESULTS_PATH = "./resultsPath"
#import requests

#session = requests.Session()

#api_key_auth = "ZmxvdzM2MDo1ZzhLdGZoSjVpbHl4TVZBNmlUTkdYRml0UlRpTVZXNQ=="

UserConfig.set_profile(
    "rui.cheng@flexcompute.com"
)  # Useful for debugging to ensure the correct account is used.
set_logging_level("INFO")

flow360.Env.prod.active()
cmap = matplotlib.colormaps['tab10']
plt.rcParams['font.size'] = 16

#flow360 configure  --profile rui.cheng@flexcompute.com --apikey "ZmxvdzM2MDo1ZzhLdGZoSjVpbHl4TVZBNmlUTkdYRml0UlRpTVZXNQ=="

def read_case_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def read_caseID(file_name):
    # Initialize an empty list to store the data
    data_array = []
    
    try:
        # Open the file in read mode
        with open(file_name, 'r') as file:
            # Read each line from the file and append to the list
            for line in file:
                # Optionally, remove any unwanted whitespace or newline characters
                data_array.append(line.strip())  # You can also modify this if you want to parse data
                
        return data_array        
    except FileNotFoundError:
        print(f"The file at {file_name} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def fetch_data(case_id):
    case = flow360.Case(case_id)
    case.results.download(total_forces=True, surface_forces=False, bet_forces=False, cfl=False)
    #case._download_file(file_name="metadata/yPlusInfo.json")
    forcename=case_id+'_total_forces_v2.csv'
    os.rename('total_forces_v2.csv', forcename)
    #case.params.to_flow360_json('flow360.json')
    return forcename

def get_data_at_last_pseudo_step(filename):
    #print("rui", filename)
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data_raw = dataframe.to_dict("list")
    data = {}
    n = len(data_raw['physical_step'])
    # initialize the dict data
    for key in data_raw.keys():
        data[key] = []

    # populate the dict data, pick the last pseudo step at the end of each physical step
    j=0
    for i in range(0, n-1):
        if data_raw['physical_step'][i] != data_raw['physical_step'][i+1]:
            #print(i)
            j+=1
            for key in data_raw.keys():
                data[key].append(data_raw[key][i])
    print("total_physical_steps=", j)
    for key in data_raw.keys():
        data[key].append(data_raw[key][n-1])
        #print(key, data[key])
    return data

def get_data_last_aver_npseudo_step(filename, npseduo):
    #print("rui", filename)
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data_raw = dataframe.to_dict("list")
    data = {}
    n = len(data_raw['physical_step'])
    #print("n=", n)
    # initialize the dict data
    for key in data_raw.keys():
        data[key] = []

    # populate the dict data, pick the last pseudo step at the end of each physical step
    j=0
    for i in range(0, n-1):
        if data_raw['physical_step'][i] != data_raw['physical_step'][i+1]:
            #print(i)
            j+=1
            for key in data_raw.keys():
                meanvalue=sum(data_raw[key][i-npseduo+1:i+1])/npseduo
                data[key].append(meanvalue)
                #data[key].append(data_raw[key][i])
                #print(key, data[key])
    for key in data_raw.keys():
        meanvalue=sum(data_raw[key][i-npseduo+1:i+1])/npseduo
        data[key].append(meanvalue)
        #print(key, data[key])            
    print("total_physical_steps=", j)

    return data


def plot_forces(folder, case, forces, coll, forcestoplot):
    labels=forcestoplot
    jrange=[0, 1, 2]
    irange=[0, 1]
    ni=len(irange)
    nj=len(jrange)
    colori=0
    fig, axs = plt.subplots(ni, nj, figsize=(28, 14),  layout='constrained')
    #formatter = ticker.FormatStrFormatter('%0.0f') # No decimal places
    for i in irange:
        for j in jrange:
            index=i*nj+j
            forcename=forcestoplot[index]
            label=labels[index]
            plot_line(i, j, coll, forces, forcename, label, axs, colori, case)
            axs[i, j].set_xlabel('alpha [deg]')
            axs[i, j].grid()          
            axs[i, j].tick_params(axis='both', labelsize=14)  
            #axs[i, j].yaxis.set_major_formatter(formatter)
            #axs[i, j].ticklabel_format(axis='y', style='sci')
    figurename=folder+"/data/"+case+"/figures/"+folder+"_forces_coeff.png"
    print(figurename)
    plt.savefig(figurename, dpi=500, bbox_inches = 'tight')
    return 0

def plot_line(i, j, coll, forces, forcename, label, axs, colori, case):

    if colori<4:
        axs[i, j].plot(coll, forces[forcename], '-', label=case, color = cmap.colors[colori])
    else:
        colorii=colori-4
        axs[i, j].plot(coll, forces[forcename], '--', label=case, color = cmap.colors[colorii])
    axs[i, j].set_ylabel(label)
    return 0  

def plot_forces_diff(folder, case, forces, coll, forcestoplot):
    #forcestoplot=['CL', 'CD', 'CFy', 'CMx', 'CMy', 'CMz']
    labels=['delta_CL', 'delta_CD', 'delta_CFy', 'delta_CMx', 'delta_CMy', 'delta_CMz']
    jrange=[0, 1, 2]
    irange=[0, 1]
    ni=len(irange)
    nj=len(jrange)
    colori=0
    fig, axs = plt.subplots(ni, nj, figsize=(28, 14), layout='constrained')
    for i in irange:
        for j in jrange:
            index=i*nj+j
            forcename=forcestoplot[index]
            label=labels[index]
            plot_line(i, j, coll, forces, forcename, label, axs, colori, case)
            axs[i, j].set_xlabel('alpha [deg]')
            axs[i, j].grid()

    figurename=folder+"/"+case+"/figures/"+folder+"_delta_forces_coeff.png"
    print(figurename)
    plt.savefig(figurename, dpi=500, bbox_inches = 'tight')
    return 0


def readtestdata():
### flap 40/37 for HLPW4    
    testAOA=[-3.803630114, 0.630306005, 2.781749964, 4.937990189, 6.033500195, 7.045030117,\
            8.113100052, 9.171830177, 10.24489975, 11.2947998, 12.35690022, 13.4066, \
            14.46520042, 15.53349972, 16.01869965, 16.53619957, 17.04520035, 18.05450058, \
            18.56760025, 19.07299995, 19.57209969, 20.54969978, 21.46619987]
    
    testCD=[0.10334, 0.116306998, 0.134581, 0.159256995,0.173684999, 0.186706007, \
            0.201723993, 0.218577996, 0.235310003, 0.249883994, 0.265890002, 0.282851994,\
            0.298635006, 0.313288003, 0.320418, 0.326321006, 0.332423002, 0.343849987, \
            0.349783003, 0.356994003, 0.361550003, 0.367354006, 0.436477989]
    testCL=[0.586673021,1.156720042,1.362380028,1.584380031,1.688420057,1.778620005,\
            1.877210021,1.977859974,2.0697999,2.149060011,2.228889942,2.312809944,\
            2.380460024,2.438509941,2.462029934,2.478869915,2.495490074,2.51967001,\
            2.524410009,2.523799896,2.514909983,2.463219881,2.318239927]
    testCMy=[-0.336261004,-0.399899989,-0.38962701,-0.381300986,-0.375851989,-0.370599985,\
             -0.363555998,-0.357874006,-0.353008986,-0.344168007,-0.338178992,-0.334307998,\
             -0.323486,-0.310099006,-0.303438008,-0.294616997,-0.287313014,-0.270090997,\
             -0.261483014,-0.25413999,-0.244862005,-0.232053995,-0.303606004]
    
    return testAOA, testCL, testCD, testCMy


def readvolantdata():
### flap 40/37 for HLPW4    
    testAOA=[1500]
    testCFz=[14.947]
    testCMz=[2.0163]
    
    return testAOA, testCFz, testCMz
    

def plot_forces_comp(folder, cases, fname, forces, coll, forcestoplot, testdata, xlabel):    
    ncase=len(cases)
    labels=forcestoplot
    jrange=[0, 1, 2]
    irange=[0, 1]
    ni=len(irange)
    nj=len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(28, 14))
    plt.subplots_adjust(wspace=0.22, hspace=0.2) 

    for ii in range(0, ncase):              
        for i in irange:
            for j in jrange:
                index=i*nj+j
                forcename=forcestoplot[index]
                label=labels[index]
                plot_line(i, j, coll, forces[ii], forcename, label, axs, ii, cases[ii])

#### adding reference test data for comparison    
    if testdata== True:              
        testAOA, testCL, testCD, testCMy=readtestdata()
        axs[0, 0].plot(testAOA, testCL, 'o', label='test', color='k')
        axs[0, 1].plot(testAOA, testCD, 'o', label='test', color='k')
        axs[1, 2].plot(testAOA, testCMy, 'o', label='test', color='k')


#    if testdata== True:              
#        testAOA, testCFz, testCMz=readvolantdata()
#        axs[0, 2].plot(testAOA, testCFz, 'o', label='test', color='k')
#        axs[1, 2].plot(testAOA, testCMz, 'o', label='test', color='k')
#        axs[0, 2].legend()
#        axs[1, 2].legend()

    axs[0, 1].set_title(folder, fontsize=20)
    axs[0, 0].legend(fontsize=16)     


    for i in [0, 1]:
        for j in [0, 1, 2]:
           #axs[i, j].set_xlabel('alpha [deg]')
           axs[i, j].set_xlabel(xlabel)
           axs[i,j].grid()

    figurename=folder+"/figures/forces/"+fname+".png"
    print("figure name=", figurename)
    plt.savefig(figurename, dpi=500, bbox_inches = 'tight')

    return 0

def list_difference(list1, list2):
    # Elements in list1 but not in list2
    diff=[0]*len(list1)
    
    for i in range(len(list1)):
        diff[i] = list1[i]-list2[i]
    return diff

def _getCaseRuntimeStats(case: flow360.Case) :
    queryUrl = f"https://admin-api.simulation.cloud/admin/jobs/support/logs/resource/{case.info.user_id}/{case.id}"

    resp = http.session.get(
        queryUrl,
        auth=api_key_auth,
    )

    #resp = session.get(
    #    queryUrl,
    #    auth=api_key_auth,
    #)    
    logpath=os.path.join(RESULTS_PATH, case.name)
    print(logpath)

    
    if not os.path.exists(os.path.join(RESULTS_PATH, case.name)):
        os.mkdir(os.path.join(RESULTS_PATH, case.name))

    zipFile = os.path.join(RESULTS_PATH, case.name, "supportLogs.zip")

    with open(zipFile, "wb") as f:
        f.write(resp.content)
    shutil.unpack_archive(zipFile, os.path.join(RESULTS_PATH, case.name))

    return os.path.join(RESULTS_PATH, case.name)

    #return _getSimulationStats(
    #    glob.glob(os.path.join(RESULTS_PATH, case.name, "*Flow360Solver*.log"))[0]
    #)

def _getSimulationStats(solverOut):
    nodeList = ["N/A"]
    numRanks = -1
    runTime = -1
    GPUList = ["N/A"]
    try:
        nodeList, numRanks, GPUList = _getNodesAndMPIRanks(solverOut)
        runTime = _getTotalRunTime(solverOut)
    except:
        pass
    simuStats = dict()
    simuStats["nodeList"] = nodeList
    simuStats["numOfRanks"] = numRanks
    simuStats["GPU"] = GPUList
    simuStats["runTimeInSeconds"] = runTime
    return simuStats

def _getTotalRunTime(filePath):
    runTime = -1
    with open(filePath, "r") as fh:
        for line in fh:
            ret = re.findall(
                "Wall clock time for time marching: ([0-9\.]+) seconds", line
            )
            if len(ret) > 0:
                assert len(ret) == 1
                runTime = float(ret[0])
    return (
        runTime  # get the last one in the log. This is to support local test reports.
    )

def getRuntimeTable(case: fl.Case, caseRef: fl.Case = None):
    caseIds = [case.id]
    stats = [_getCaseRuntimeStats(case)]
    print(stats[0])
    return stats[0]

def _getClusterFromNodeName(nodeList):
    clusters = set()
    for nodeName in nodeList:
        if re.match("cell[0-9]+", nodeName):
            names = re.findall("cell[0-9]+", nodeName)
            assert len(names) == 1
            clusters.add(names[0])
        elif re.match("cloud-google-worker", nodeName):
            clusters.add("gcloud")
        elif re.match("a5k-[0-9]{3}", nodeName):
            clusters.add(nodeName)
        elif re.match("h200\-[0-9]{3}", nodeName):
            names = re.findall("h200\-[0-9]{3}", nodeName)
            assert len(names) == 1
            clusters.add(names[0])
        elif re.match("b200\-[0-9]{3}", nodeName):
            names = re.findall("b200\-[0-9]{3}", nodeName)
            assert len(names) == 1
            clusters.add(names[0])
        else:
            log.warning("Unknown node name: {}".format(nodeName))
            clusters.add(nodeName)
    return clusters

def main():     
    forcearray = {}
    allforces={}
    diff={}
    diffs={}
     
### reference case is used for get all keys
    case_id = 'case-9bab22dd-cb70-4910-92b2-d332a5f99dc1'
    cvsforces_file = fetch_data(case_id)
    forces = get_data_at_last_pseudo_step(cvsforces_file)
    for key in forces.keys():
        #print(key)
        diffs[key] =[]
        allforces[key]=[]

    #config_file="./config_files/sliding_XV15_isorotor_config.json";
    #config_file="./config_files/release25.6_test_isorotor_config_0708_2025.json"
    #config_file="./config_files/release25.6_dufour_config_0728_2025.json"
    #config_file="./config_files/release25.6_eVTOL_config_0728_2025.json"
    config_file="./config_files/config.json"


    config = read_case_config(config_file)
    rootfolder = config["rootfolder"]
    casenames = config["casenames"]
    releases = config["releases"]
    subcases = config["subcases"]
    datafileexist = config["datafileexist"]
    AOAs = config["AOAs"]
    figure_extname = config["figure_extname"]
    testdata = config["testdata"]
    rotorflag = config["rotorflag"]
    wholeplane = config["wholeplane"]
    xlabel = config["xlabel"]
    nperiod = config["nperiod"]
    scales = config["scales"]    
    npseduos= config["npseduos"]

##################################################################################################################################  
    #### defualt force to plot
    forcestoplot=['CL', 'CD', 'CFy', 'CMx', 'CMy', 'CMz']

    if rotorflag == True:
        forcestoplot=['CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz']
    if wholeplane==True:
        #forcestoplot=['CL', 'CD', 'CDSkinFriction', 'CDPressure', 'CLSkinFriction', 'CMy']     
        forcestoplot=['CL', 'CD', 'CDSkinFriction', 'CDPressure', 'CMx', 'CMy']       

    cases=[]
    ncases=len(casenames)

    for i in range(0, ncases):
        scale=scales[i]
        np=nperiod[i]
        npseduo=npseduos[i]

        forcearray={}
        for key in forces.keys():
            forcearray[key] = []
        
        #### create folder for each case
        cases.append(casenames[i]+'_'+releases[i])
        path = os.path.join(rootfolder, "data", casenames[i]+'_'+releases[i])
        print("##############################################################")
        print("this part read in data of case:", i)
        print("PATH=", path)

        ## create folder for each case figures
        figurepath=os.path.join(path, "figures")
        try:
            os.makedirs(path, exist_ok=False)
            os.makedirs(figurepath, exist_ok=False)
            print(f"Folder '{casenames[i]}' created successfully at {path}")
        except OSError as error:
            print(f"Error creating folder '{casenames[i]}': {error}")

        ### if first time to run the case, fetch data from flow360        
        first=datafileexist[i]
        ### read case ID from file
        caseID_file = rootfolder+'/caseIDfiles/'+casenames[i]+'_'+releases[i]+'_'+subcases[i]+'.txt' 
        print("caseID_file=", caseID_file)
        case_ids = read_caseID(caseID_file)

        if case_ids is not None:
            print("Data read from the following case_ids:")
            print(case_ids)

        for case_id in case_ids:
            if first==1 :
                ### if first time to run the case, fetch data from flow360   
                cvsforces_file = fetch_data(case_id)
                newname=path+'/'+case_id+'_total_forces_v2.csv'
                #print(newname)
                shutil.move(cvsforces_file, newname)
            else :
                ### if data file already exist, just read the data
                newname=path+'/'+case_id+'_total_forces_v2.csv'   

            #times=_getCaseRuntimeStats(flow360.Case(case_id))  

            #print(times)

            #filepath=times+"/casePipeline.Flow360Solver."+case_id+".log"

            #print("filepath=", filepath)
            #times2=_getTotalRunTime(filepath)  
            #print(times2)
                      
            #getRuntimeTable(flow360.Case(case_id))
            #stop

            #print(table)

            #forces = get_data_at_last_pseudo_step(newname)
            #### if np >2, it is unsteady case, take the average of the last np physical steps, 
            #### if np=1, it is steady case, take the last pseudo step in the subroutine
            if np<2:
                forces = get_data_last_aver_npseudo_step(newname, npseduo)
                #print("npseduo=", npseduo)
                ### Take the average of the last np pseudo steps
                for key in forces.keys():
                    forcearray[key].append(forces[key][-1])
                    #print(key, forces[key][-1])
            else:
                forces = get_data_at_last_pseudo_step(newname)
                ### Take the average of the last np physical steps
                for key in forces.keys():                    
                    meanvalue=sum(forces[key][-(np+1):-1])/np/scale
                    #print(meanvalue, sum(forces["CL"][-(np+1):-1]), forces["CL"][-(np+1):-1])
                    forcearray[key].append(meanvalue)
                    #forcearray[key].append(forces[key][-1])            
        #print("Rui", forcearray)
        
        ### plot forces for each case
        plot_forces(rootfolder, cases[i],  forcearray, AOAs, forcestoplot)
        allforces[i]=forcearray

    print("###########################################################")
    print("print out CL for data verify")
    for i in range(0, ncases):

        print(i, "CL", allforces[i]["CL"])
        print(i, "CD", allforces[i]["CD"])

    print("###########################################################")

    ### plot forces comparison between all cases
    figurename=rootfolder+"_forces_coeff_compare"+figure_extname 
    plot_forces_comp(rootfolder, cases, figurename, allforces, AOAs, forcestoplot, testdata, xlabel)

    ### plot forces difference between all cases, one by one
    diffnames=[]
    for i in range (0, ncases-1):
        diff={}
        for key in forces.keys():
            diff[key]=[]
        for key in forces.keys():
            diff[key] = list_difference(allforces[i][key], allforces[i+1][key])

        diffname=cases[i]+'-'+cases[i+1]
        diffs[i]=diff
        diffnames.append(diffname)    
    #testdata=False
    figurename=rootfolder+"_forces_coeff_diff"+figure_extname 
    plot_forces_comp(rootfolder, diffnames, figurename, diffs, AOAs, forcestoplot, testdata, xlabel)

    return 0
if __name__ == '__main__':
    main()
