import pandas as pd
from rdkit import Chem
import csv
import requests
from tqdm import tqdm


url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES,title/CSV"
headers = {"content-type":"application/x-www-form-urlencoded"}
files = ["training", "val", "test"]

for file in files:
    df = pd.DataFrame(columns=["CID", "CanonicalSMILES", "Title", "Description"])
    with open(f"data/{file}.txt") as f:
        for line in tqdm(csv.reader(f, delimiter='\t')):
            cid = line[0]
            desc = line[-1]

            data={"cid":cid}
            res = requests.post(url, data=data, headers=headers)
            
            smi = res.text.split('"')[7]
            mol = Chem.MolFromSmiles(smi)
            if mol.GetNumBonds() > 1:
                try:
                    # there may be some Depositor-Supplied Synonyms of title
                    df = pd.concat([df, pd.DataFrame([{"CID": cid, "CanonicalSMILES": smi, "Title": res.text.split('"')[9], "Description": desc}])], ignore_index=True)
                except:
                    continue

    # print(len(df))
    df.to_csv(f"data/{file}.csv", index=None)