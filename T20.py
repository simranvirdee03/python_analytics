import pandas as pd
import json

with open("C:/Users/Harpreet Singh/Desktop/t20_json_files/t20_wc_match_results.json") as f:
    data =json.load(f)
df_match=pd.DataFrame(data[0]['matchSummary'])
df_match.rename({'scorecard': 'match_id'}, axis= 1, inplace=True)

with open("C:/Users/Harpreet Singh/Desktop/t20_json_files/t20_wc_batting_summary.json") as f:
    data=json.load(f)
    all_records=[]

    for rec in data:
        all_records.extend(rec['battingSummary'])
df_batting=pd.DataFrame(all_records)
df_batting.head()
df_batting["out/not_out"]=df_batting.dismissal.apply(lambda x: "out" if len(x)>0 else "not_out")
df_batting.drop(columns=["dismissal"],inplace=True)
df_batting.head(2)
df_batting['batsmanName']=df_batting['batsmanName'].apply(lambda x: x.replace('a€',' '))
df_batting['batsmanName']=df_batting['batsmanName'].apply(lambda x: x.replace('\xa0',' '))
#print(df_batting.head(3))

match_ids_dict={}
for index, row in df_match.iterrows():
    key1=row['team1']+' Vs '+row['team2']
    key2=row['team2']+' Vs '+row['team1']

    match_ids_dict[key1]=row["match_id"]
    match_ids_dict[key2]=row["match_id"]

#print(match_ids_dict["Namibia Vs Sri Lanka"])
df_batting['mach_id']=df_batting['match'].map(match_ids_dict)
print(df_batting)
df_batting.to_csv("C:/Users/Harpreet Singh/Desktop/t20_json_files/002.csv",index=False)