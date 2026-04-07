import os
import glob
import re
import numpy as np
import importlib
import sys
import scipy.stats as st
import matplotlib.pyplot as plt
'''
#Testidata
#Aortta data
aorta=[0,124,886,5406,29009,84960,90750,78030,57405,
42104,34094,27709,23290,22109,21168,18760,16693,
15754,14843,14245,13780,13292,12567,12291,11974]

#aivo data
brain=[0,131,120,116,271,1984,4982,8035,10641,
12466,13561,14294,14700,15073,15312,15330,15338,
15308,15131,14855,14643,14123,13874,13473,13141]
'''
#aika
time=[3,8,13,18,23,28,33,38,
    43,48,53,58,63,68,75,85,
    95,110,130,150,175,205,235,265]

# ----------------------------------------
# Polku datakansioon
# ----------------------------------------
DATA_DIR = r"C:\Users\Valtteri\Documents\Mallinnnusprojekti/Ajettavat_ULA_datat"

# Polku analyysityökaluihin
ANALYSIS_TOOLS_DIR = r"C:\Users\Valtteri\Documents\Mallinnnusprojekti/RUN_ajettavat_ULA_koodit"

# ----------------------------------------
# Luetaan analyysityökalut (.py)
# ----------------------------------------
tools = [
    f[:-3] for f in os.listdir(ANALYSIS_TOOLS_DIR) #eli haetaan koodit niiden sijainnista
    if f.endswith(".py") and not f.startswith("__") # ne jotka loppuu .py ja ei ala __
]

print("\n>>> LÖYDETYT ANALYYSITYÖKALUT:")
for t in tools:
    print(" -", t)

# --------------------------
# LISÄTTY: lisää analyysityökalujen kansio import-polkuun
sys.path.insert(0, ANALYSIS_TOOLS_DIR) #varmistaa että haetaan juuri oikeasta kansiosta

# --------------------------
# Regex datan parsimiseen
pattern = re.compile(
    r"([A-Za-z]+)\s*\(([\d\.]+)\s*mm3\):\s*\[([^\]]+)\]",
    re.MULTILINE
)

# --------------------------
# Tietorakenteet
brain_data = []      # organ_data["Lungs__dataset1"] = numpy array
lungs_data = []      # organ_data["Lungs__dataset1"] = numpy array
spleen_data = []     
aorta_data = []      

organ_data = []
organ_volume = []



# --------------------------
# 1) LUETAAN KAIKKI DATA
for tiedosto in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
    with open(tiedosto, "r") as f:
        content = f.read() # on kaikki datatiedostot

    matches = pattern.findall(content)
    dataset_name = os.path.basename(tiedosto).replace(".txt", "")
    #print(matches)
    #print('-'*40)
    #print(dataset_name)
    for organ, vol, arr_str in matches:
        vol = float(vol)
        vol = vol/1000000
        numbers = np.fromstring(arr_str, sep=' ')
        key = f"{organ}__{dataset_name}"

        match = re.search(r'\d+', key)
        numero = int(match.group())
        numero = int(str(numero).strip())
        #print(organ,numero,numbers,vol)
        if organ.lower() =='lungs':
            l = [numero,numbers,vol]
            lungs_data.append(l)
            #print(lungs_data)
        elif organ.lower() =='brain':
            li = [numero,numbers,vol]
            brain_data.append(li)
        elif organ.lower() =='spleen':
            lis = [numero,numbers,vol]
            spleen_data.append(lis)
        elif organ.lower() =='aorta':
            lista = [numero,numbers,vol]
            aorta_data.append(lista)
    organ_data = [lungs_data, brain_data, spleen_data]
  

# --------------------------
'''
# Tulostetaan löydetyt tiedot
print("\n>>> LÖYDETYT TIEDOT:")
for organ in organ_data:
    print(f"{organ}: tilavuus={organ_volume[organ]} mm³, datapisteitä={len(organ_data[organ])}")
'''
# --------------------------
# 2) AJETAAN JOKAINEN ANALYYSITYÖKALU KAIKELLA DATALLA
#print(tools)
#Elimet = {1:'lungs',2:'brain',3:'spleen'}
comparison_data = {}
for tool in tools:
    print(f"\n============================")
    print(f"   AJETAAN TYÖKALU: {tool}")
    print(f"============================")
    
    module = importlib.import_module(tool)  # työkalu moduuli
    #print(module.run(brain, 1293.9384841918945, aorta, 5100, time))
    print('-----------')
    # Käydään kaikki datasetit läpi # ORGAN data = lungs, brain, spleen
    
    laskuri = 0
    
    L = [0,1,2]
    
    Elimet = ['lungs','brain','spleen']
    for elin, ELI in zip(organ_data, Elimet):
        tuloste = []
        virhe = []
        arvot_k1 = []
        arvot_k2 = []
        
        for j in range(len(elin)):
            if elin[j][0] == aorta_data[j][0]:
                
                if hasattr(module, "run"):
                            # MUUTTUNUT: annetaan nyt 5 muuttujaa
                    #print(module.run(brain, 1293.9384841918945, aorta, 5100, time))
                    
                    error, k1, k2 =  module.run(elin[j][1], elin[j][2], aorta_data[j][1], aorta_data[j][2], time)
                    #print('tulokset datasetille',j,':','virhe',error, 'k1', k1,'k2', k2)
                    virhe.append(error)
                    arvot_k1.append(np.float64(k1))
                    arvot_k2.append(np.float64(k2))
                    #print(virhe, arvot_k1, arvot_k2)
                    
                else:
                    print(f"!!! VAROITUS: työkalussa {tool} ei ole funktiota run() !!!")
            
            else:
                print('datan indeksit ei täsmää')
        tuloste.append(virhe)
        tuloste.append(arvot_k1)
        tuloste.append(arvot_k2)
        
        Elimet = ['lungs','brain','spleen']
        '''
        for i in range(len(Elimet)):
            print(len(tuloste))
            print(tuloste[i])
        '''
        virheet_lungs = []
        virheet_brain = []
        virheet_spleen = []
        #print(tuloste)
        Halutut = ['Erotusten neliö summa','virhe k1','virhe k2']
        print(tool)
        print(ELI)
        
        for i,H in zip(tuloste,Halutut):
                #print(len(tuloste))
                if tuloste.index(i)==0:
                    N = len(i)
                    mean = np.mean(i) #KA
                    mean_piste = mean/N
                    std = np.std(i, ddof=1) #keskihajonta
                    t = st.t.ppf(0.975, N-1)
                    conf_interval = (mean - t*std/np.sqrt(N), mean + t*std/np.sqrt(N)) #Luottamusväli
                        
                    print('Virhe:',H)
                    print('KA:',mean,'yksittäisen pisteen KA virhe:',mean_piste,'keskihajonta:',std)
                    #print('K1 ja k2 välinen korrelaatio:',np.corrcoef(k1_lista, k2_lista)[0,1])
                    print('-'*40)
                else:
                    N = len(i)
                    mean = np.mean(i) #KA
                    mean_piste = mean/N
                    std = np.std(i, ddof=1) #keskihajonta
                    t = st.t.ppf(0.975, N-1)
                    conf_interval = (mean - t*std/np.sqrt(N), mean + t*std/np.sqrt(N)) #Luottamusväli
                        
                    print('Virhe:',H)
                    print('KA:',mean,'keskihajonta:',std)
                    #print('K1 ja k2 välinen korrelaatio:',np.corrcoef(k1_lista, k2_lista)[0,1])
                    print('-'*40)
        print('-'*40)
        # --- 2. STORE DATA ---
        if ELI not in comparison_data:
            comparison_data[ELI] = {}
        
        # Save the results for this tool and organ to plot later
        comparison_data[ELI][tool] = tuloste
        # --- PLOTTING SNIPPET END ---      
    #print('virhe','k1','k2',)
# --- 3. FINAL COMPARISON PLOTTING ---
print("\n>>> GENERATING COMPARISON GRAPHS...")

for organ, tools_dict in comparison_data.items():
    # Create 3 subplots for this Organ (Error, k1, k2)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plot_titles = ['Sum Squared Error', 'k1 Value', 'k2 Value']
    
    # Loop through the 3 metrics
    for metric_idx, ax in enumerate(axes):
        # Plot EVERY tool on the same axis
        for tool_name, data_list in tools_dict.items():
            metric_data = data_list[metric_idx]
            x_vals = range(len(metric_data))
            
            # Plot line for this tool
            ax.plot(x_vals, metric_data, marker='o', markersize=4, linestyle='-', alpha=0.7, label=tool_name)
        
        ax.set_title(plot_titles[metric_idx])
        ax.set_xlabel('Dataset Number')
        ax.set_ylabel('Value')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend() # Shows which color belongs to which tool
        #ax.set_yscale('log')
        
    plt.suptitle(f"Method Comparison for: {organ.upper()}", fontsize=16)
    plt.tight_layout()
    
    filename = f"comparison_{organ}.png"
    plt.savefig(filename)
    print(f" - Saved {filename}")
    plt.close()
# --- 4. DATA TABLES (Mean +/- Std) ---
print("\n>>> GENERATING SUMMARY TABLES...")

for organ, tools_dict in comparison_data.items():
    # Prepare data for the table
    cell_text = []
    row_labels = []
    col_labels = ['Sum Squared Error', 'k1 Value', 'k2 Value']
    
    for tool_name, data_list in tools_dict.items():
        row_labels.append(tool_name)
        row_data = []
        
        # Calculate stats for Error, k1, and k2
        for metric_data in data_list:
            avg = np.mean(metric_data)
            std = np.std(metric_data, ddof=1) # Sample standard deviation
            
            # Format: "Mean ± Std"
            # Use scientific notation for Error if it's huge, normal for others
            if avg > 1000:
                row_data.append(f"{avg:.2e} \n± {std:.2e}")
            else:
                row_data.append(f"{avg:.4f} \n± {std:.4f}")
        
        cell_text.append(row_data)

    # Create a figure to draw the table on
    fig, ax = plt.subplots(figsize=(14, len(row_labels) * 1.2 + 2)) # Dynamic height
    
    # Hide axes (we just want the table)
    ax.axis('off')
    ax.axis('tight')
    
    # Draw the table
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 4) # Stretch height for readability
    
    plt.title(f"Statistical Summary: {organ.upper()} (Mean ± Std Dev)", fontsize=14, weight='bold')
    
    filename = f"table_{organ}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f" - Saved {filename}")
    plt.close()
'''

            '''

            #print(laskuri,"Tulostaa datasetin",elin[j][0],tulokset)
            


