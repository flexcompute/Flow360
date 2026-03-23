import flow360
import math, json, pandas
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import json


flow360.Env.preprod.active()
cmap = matplotlib.colormaps['tab10']
plt.rcParams['font.size'] = 18

def read_case_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def read_file_line(file_path, iline):
    linelist=[]
    with open(file_path, 'r') as file:
        lines = file.readlines()   
        linelist.append(lines[iline].strip())
        #return lines[iline].strip()
        return linelist
        
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

def fetch_totforce(case_id, totforce_flag, residual_flag):
    case = flow360.Case(case_id)
    case.results.download(total_forces=totforce_flag, nonlinear_residuals=residual_flag, surface_forces=False, bet_forces=False, cfl=False)
    if totforce_flag==True:
        forcename=case_id+'_total_forces_v2.csv'
        os.rename('total_forces_v2.csv', forcename)
    if residual_flag==True:
        forcename=case_id+'_nonlinear_residual_v2.csv'
        os.rename('nonlinear_residual_v2.csv', forcename)
    return forcename

def get_convergence_data(filename):
    #print("rui", filename)
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data = dataframe.to_dict("list")
    return data

def get_data_at_last_pseudo_step(filename):
    #print("rui", filename)
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data_raw = dataframe.to_dict("list")
    data = {}
    n = len(data_raw['physical_step'])
    print("nstep=", n)
    # initialize the dict data
    for key in data_raw.keys():
        data[key] = []

    # populate the dict data, pick the last pseudo step at the end of each physical step
    j=0
    for i in range(0, n-1):
        if data_raw['physical_step'][i] != data_raw['physical_step'][i+1]:
            #print(i)
            j+=1;
            for key in data_raw.keys():
                data[key].append(data_raw[key][i])
    for key in data_raw.keys():
        data[key].append(data_raw[key][n-1])
        #print(key, data[key])
    return data

def plot_convergence_comparison(folder, cases, forces, AOA, step,  forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel):

    forcenames=forcestoplot
    jrange=[0, 1, 2]
    irange=[0, 1]
    ni=len(irange)
    nj=len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(24, 14))

    print("this is for plot case of", xlabel, ": ", AOA, ", step ", step)
    ncases=len(cases)
    maxphstep=max(forces[0]['physical_step'])
    lim = [[None, None] for _ in range(ni * nj)]
    for icase in range (0, ncases):
        case=cases[icase]
        colori=icase
        coll=forces[icase]['pseudo_step']
        for i in irange:
            for j in jrange:
                index=i*nj+j
                forcename=forcestoplot[index]
                forcename=forcenames[index]
                plot_line(i, j, coll, forces[icase], forcename, axs, colori, case, residual_flag)
                vmin = min(forces[icase][forcename])
                vmax = max(forces[icase][forcename])
                lim[index][0] = vmin if lim[index][0] is None else min(lim[index][0], vmin)
                lim[index][1] = vmax if lim[index][1] is None else max(lim[index][1], vmax)
                #axs[i, j].grid()
                if(maxphstep==0):
                   axs[i, j].set_xlabel('pseudo_step')
                else:
                   axs[i, j].set_xlabel('physical_step')
                axs[i, j].set_ylabel(forcename)
                #axs[i,j].grid()

    # Apply padding to lim and set axis limits
    for i in irange:
        for j in jrange:
            index=i*nj+j
            vmin = lim[index][0]
            vmax = lim[index][1]
            if totforce_flag:
                span = vmax - vmin
                lim[index][0] = vmin - 0.1 * span
                lim[index][1] = vmax + 0.1 * span
            else:  # residual
                lim[index][0] = vmin / 10
                lim[index][1] = vmax * 10
            axs[i, j].set_ylim(lim[index][0], lim[index][1])             
              
    #title=folder+" "+"AOA="+str(AOA)
    title=folder+" "+xlabel+"="+str(AOA)
    axs[0, 1].set_title(title, fontsize=20)
    axs[0, 0].legend(fontsize=16, loc='upper center')   

    for i in irange:
        for j in jrange:
            axs[i,j].grid()
    if totforce_flag==True:
        figurename=folder+"/figures/"+figure_extname+"force_history_"+xlabel+str(AOA)+"_last"+str(step)+"step.png"
    if residual_flag==True:
        figurename=folder+"/figures/"+figure_extname+"residual_"+xlabel+str(AOA)+"_last"+str(step)+"step.png"
    
    print("figurename:", figurename)
    plt.savefig(figurename, dpi=500, bbox_inches = 'tight')

    plt.close()

    return 0


def plot_convergence_comparison_range(folder, cases, forces, AOA, step, forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel):
    jrange=[0, 1, 2]
    irange=[0, 1]
    ni=len(irange)
    nj=len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(24, 14))

    print("this is for plot case of range", xlabel, ": ", AOA, " step ", step)
    ncases=len(cases)
    maxphstep=max(forces[0]['physical_step'])
    coll=[]
    lim=np.zeros((len(forcestoplot), 2))

    # Pre-pass: compute lim from the last data point of each case
    for i in irange:
        for j in jrange:
            index=i*nj+j
            forcename=forcestoplot[index]
            last_values = [forces[icase][forcename][-1] for icase in range(ncases)]
            vmin = min(last_values)
            vmax = max(last_values)
            if totforce_flag:
                span = vmax - vmin
                lim[index][0] = vmin - 2 * span
                lim[index][1] = vmax + 2 * span
            else:  # residual
                lim[index][0] = vmin / 10
                lim[index][1] = vmax * 10

    for icase in range (0, ncases):
        case=cases[icase]
        colori=icase
        coll = forces[icase]['pseudo_step']
        for i in irange:
            for j in jrange:
                index=i*nj+j
                forcename=forcestoplot[index]
                plot_line(i, j, coll, forces[icase], forcename, axs, colori, case, residual_flag)
                axs[i, j].set_ylabel(forcename)
                axs[i, j].set_ylim(lim[index][0], lim[index][1])

                if icase==0:
                    axs[i, j].grid()
                if(maxphstep==0):
                   axs[i, j].set_xlabel('pseudo_step')
                else:
                   axs[i, j].set_xlabel('physical_step')

                #axs[i,j].grid()             
              
    #title=folder+" "+"AOA="+str(AOA)
    title=folder+" "+xlabel+"="+str(AOA)

    axs[0, 1].set_title(title, fontsize=20)
    axs[0, 1].legend(fontsize=14, loc='upper center')  

    #for i in irange:
    #    for j in jrange:
    #        axs[i,j].grid()
    if totforce_flag==True:
        figurename=folder+"/figures/"+figure_extname+"range_force_history_"+xlabel+str(AOA)+"_last"+str(step)+"step.png"
    if residual_flag==True:
        figurename=folder+"/figures/"+figure_extname+"range_residual_"+xlabel+str(AOA)+"_last"+str(step)+"step.png"
    
    print("figurename:", figurename)
    plt.savefig(figurename, dpi=500, bbox_inches = 'tight')

    plt.close()

    return 0


def plot_line(i, j, coll, forces, forcename, axs, colori, case, residual_flag):
    if residual_flag == False:
        axs[i, j].plot(coll, forces[forcename], '-', label=case, color = cmap.colors[colori])
        axs[i, j].set_ylabel(forcename)
    else:
        axs[i, j].semilogy(coll, forces[forcename], '-', label=case, color = cmap.colors[colori])
        axs[i, j].set_ylabel(forcename)
    return 0


def extract_unsteady_convergence(ncases, forcesarray, steppert):
 
    forces = {}
    temp ={}
    for key in forcesarray[0].keys():
        forces[key] = []    
        temp[key]=[]  

    ### calculate how many physical step to be plot 
    minlength=max(forcesarray[0]['physical_step'])-min(forcesarray[0]['physical_step'])    
    for i in range(0, ncases):
        maxphstep=max(forcesarray[i]['physical_step'])
        minphstep=min(forcesarray[i]['physical_step'])
        minlength=min(minlength, maxphstep-minphstep)
    complength=int(steppert*minlength)
    
    ### extract data 
    for i in range(0, ncases):     
        maxphstep=max(forcesarray[i]['physical_step'])
        minphstep=min(forcesarray[i]['physical_step'])
        maxpsstep=max(forcesarray[i]['pseudo_step'])

        if (abs(steppert-1)<1E-4):
            beginstep=minphstep
        else:
            beginstep=maxphstep-complength

        beginindex=forcesarray[i]['physical_step'].index(beginstep)
        endindex=forcesarray[i]['physical_step'].index(maxphstep)-1

        for key in forcesarray[i].keys():
            temp[key]=forcesarray[i][key][beginindex:endindex]

        lentemp=len(temp['pseudo_step'])
        if (abs(steppert-1)>1E-4):
            initial_phstep=temp['physical_step'][0]
            for itemp in range(0, lentemp):
                temp['physical_step'][itemp]=temp['physical_step'][itemp]-initial_phstep

        
        ## change pseudo_step for plot only, to digitals
        for ii in range(0, lentemp):
            temp['pseudo_step'][ii]=temp['physical_step'][ii]+temp['pseudo_step'][ii]/maxpsstep
        forces[i]=temp
        temp={}
        for key in forcesarray[0].keys():
            temp[key]=[]
        
    return forces

def main():

    #config_file="./config_files/release25.6_test_isorotor_config_0728_2025.json"
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
    SST_flags = [bool(f) for f in config.get("SSTFlag", [False] * len(casenames))]
    # Pad with False if list is shorter than number of cases
    SST_flags += [False] * (len(casenames) - len(SST_flags))

############################################################################################################
############################################################################################################
    for totforce_flag, residual_flag in [(True, False), (False, True)]:
        nfirst=len(datafileexist)
        if totforce_flag==True:
            datafileexist=[0]*nfirst
        if residual_flag==True:
            datafileexist=[1]*nfirst
        print(f"Running with totforce_flag={totforce_flag}, residual_flag={residual_flag}")

        ### the default comparison
        forcestoplot=['CL', 'CD', 'CFy', 'CMx', 'CMy', 'CMz']
        if totforce_flag== True:
            if rotorflag == True:
                forcestoplot=['CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz']
            residual_flag=False

        if residual_flag== True:
            totforce_flag= False
            # forcestoplot is shared across all cases on the same plot axes;
            # use SST_flags[0] as representative — cases in the same group should use the same turbulence model
            if SST_flags[0]:
                forcestoplot=['0_cont', '1_momx', '2_momy', '4_energ', '5_k', '6_omega']
            else:
                forcestoplot=['0_cont', '1_momx', '2_momy', '3_momz', '4_energ', '5_nuHat']
        else:
            totforce_flag= True
            if rotorflag == True:
                forcestoplot=['CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz']
                

        try:
            os.makedirs(rootfolder, exist_ok=False)
            print(f"Folder created successfully for {rootfolder}")
        except OSError as error:
            print(f"Error creating folder '{rootfolder}': {error}")    

        try:
            fpath=os.path.join(rootfolder, "figures")
            os.makedirs(fpath, exist_ok=False)
            print(f"Folder created successfully for {fpath}")
        except OSError as error:
            print(f"Error creating folder '{fpath}': {error}")    

        ###############################################################################################################################
        ### this part read the case ID
        caseIDarray={}
        ncases=len(casenames)
        nAOAs=len(AOAs)

        cases=[]
        for i in range(0, ncases):      
            cases.append(casenames[i]+'_'+releases[i])
            path = os.path.join(rootfolder, "data", casenames[i]+'_'+releases[i])
            print("path=", path)
            figurepath=os.path.join(path, "figures")
            try:
                os.makedirs(path, exist_ok=False)
                figurepath=os.path.join(path, "figures")
                #print(figurepath)
                os.makedirs(figurepath, exist_ok=False)
                print(f"Folder '{casenames[i]}' created successfully at {path}")
            except OSError as error:
                print(f"Error creating folder '{casenames[i]}': {error}")

            caseID_file = rootfolder+'/caseIDfiles/'+casenames[i]+'_'+releases[i]+'_'+subcases[i]+'.txt'
            print("caseID_file=", caseID_file)
            case_ids = read_caseID(caseID_file)        
            caseIDarray[i]=case_ids

        #####################################################################################################################
        ### This part read the cases and compare the force convergence 
        #case_id = 'case-3e73ae01-e21c-489c-b0f4-4f66d617d882'
        case_id = caseIDarray[0][0]
        cvsforces_file = fetch_totforce(case_id, totforce_flag, residual_flag)
        forces = get_convergence_data(cvsforces_file)   

        for iaoa in range (0, nAOAs):
            forcearray = {}
            unsteadyforces = {}
            for key in forces.keys():
                forcearray[key] = []
                unsteadyforces[key]=[]
            
            maxphstep=0
            for icase in range (0, ncases):
                scale=scales[icase]
                path = os.path.join(rootfolder, "data", casenames[icase]+'_'+releases[icase])
                case_id=caseIDarray[icase][iaoa]
                print("icase, iaoa, case_id: ", icase, iaoa, case_id )            
                first=datafileexist[icase]
                if first==1:
                    cvsforces_file = fetch_totforce(case_id, totforce_flag, residual_flag)
                    if totforce_flag==True:
                        newname=path+'/'+case_id+'_total_forces_v2.csv'                
                        shutil.move(cvsforces_file, newname)
                    if residual_flag==True:
                        newname=path+'/'+case_id+'_nonlinear_residual_v2.csv'                
                        shutil.move(cvsforces_file, newname)
                else :
                    if totforce_flag==True:
                        newname=path+'/'+case_id+'_total_forces_v2.csv'          
                    if residual_flag==True:
                        newname=path+'/'+case_id+'_nonlinear_residual_v2.csv'
                #print(newname)

                forces = get_convergence_data(newname)
                if totforce_flag==True:
                    for key in forcestoplot:
                        nitem=len(forces[key])
                        for iitem in range(0, nitem):
                            forces[key][iitem]=forces[key][iitem]/scale
                            
                forcearray[icase]=forces   
                

                #maxphstep=max(maxphstep, max(forces['physical_step']))
                maxphstep=max(forces['physical_step'])
                #minphstep=min(forces['physical_step'])

                
            AOA=AOAs[iaoa]
            if maxphstep==0:
                print("###############################################################################")  
                print("the compared cases are steady cases")      
                plot_convergence_comparison(rootfolder, cases, forcearray, AOA, maxphstep, forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel )    
                plot_convergence_comparison_range(rootfolder, cases, forcearray, AOA, maxphstep, forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel )              
            else:
                print("###############################################################################")
                print("the compared cases are unsteady cases, iaoa", iaoa)        
                plotscales=[0.01, 0.1, 1.0]
                for plotscale in plotscales:
                    unsteadyforces=extract_unsteady_convergence(ncases, forcearray, plotscale)  
                    if (abs(plotscale-1)<1E-3):
                        extractstep=maxphstep
                    else:
                        extractstep=max(unsteadyforces[0]['physical_step'])
                    plot_convergence_comparison(rootfolder, cases, unsteadyforces, AOA, extractstep, forcestoplot, totforce_flag, residual_flag , figure_extname, xlabel)
                    plot_convergence_comparison_range(rootfolder, cases, unsteadyforces, AOA, extractstep, forcestoplot, totforce_flag, residual_flag , figure_extname, xlabel)
                    if testdata==True:
                        if rotorflag==True:
                            testname=path+'/testdata/rotor_'+xlabel+str(AOA)+'_last'+str(extractstep)+'step_scale'+str(plotscale)+'.csv'
                        else:
                            testname=path+'/testdata/wing_'+xlabel+str(AOA)+'_last'+str(extractstep)+'step_scale'+str(plotscale)+'.csv'
                        print("testname=", testname)
                        if os.path.exists(testname):
                            testforces = get_data_at_last_pseudo_step(testname)
                            plot_convergence_comparison(path, cases, [testforces], AOA, extractstep, forcestoplot, totforce_flag, residual_flag , figure_extname, xlabel)    
            
if __name__ == '__main__':
    main()
