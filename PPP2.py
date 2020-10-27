import pandas as pd
import numpy as np
import matplotlib

#matplotlib.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("C:/Users/DELL/Desktop/Videos/totorial/machine leaning/"
                 "py-master/ML/14_naive_bayes/Bengaluru_House_Data.csv")
pop = df.groupby("area_type")["area_type"].agg("count")
df2 = df.drop(["area_type","society","balcony", "availability",], axis="columns")
df3 = df2.dropna()
pop = df3.isnull().sum()
df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ") [0]))

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
bob = df3[~df3["total_sqft"].apply(is_float)].head(10)

def convert_sqft_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df4 = df3.copy()

df4["total_sqft"] = df4["total_sqft"].apply(convert_sqft_to_num)

df5 = df4.copy()
df5["price_per_sqft"] = df5["price"]*100000/df5["total_sqft"]

df5.location = df5.location.apply(lambda x: x.strip())
location_stat = df5.groupby("location")["location"].agg("count").sort_values(ascending=False)

location_stat_less_then_10 = (location_stat[location_stat<=10 ])


df5.location = df5.location.apply(lambda x: "other" if x in location_stat_less_then_10 else x )

less = df5[df5.total_sqft/df5.bhk < 300]
df6 = df5[~(df5.total_sqft/df5.bhk < 300)]

def remove_pp(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-sd)) & (subdf.price_per_sqft<=(m+sd))]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True )
    return df_out

df7 = remove_pp(df6)
#print(df7)

def renove_bhk(df):
    exclude_index = np.array([])
    for location, location_df in df.groupby("location"):
        bhk_stat = {}
        for bhk,  bhk_df in location_df.groupby("bhk"):
            bhk_stat[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std" : np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for bhk,  bhk_df in location_df.groupby("bhk"):
            stat = bhk_stat.get(bhk-1)
            if stat and stat["count"]>5:
                exclude_index = np.append(exclude_index,bhk_df[bhk_df.price_per_sqft<(stat["mean"])].index.values)
    return df.drop(exclude_index, axis="index")

df8 = renove_bhk(df7)

df9 =  (df8[df8.bath<df8.bhk+2])

df10 = df9.drop(["size", "price_per_sqft"], axis= "columns")

dummies = pd.get_dummies(df10.location)

df11 = pd.concat([df10, dummies.drop("other", axis="columns")], axis="columns")

df12 = df11.drop("location", axis= "columns")
x = df12.drop("price", axis= "columns")
y = df12.price

from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)
#to check the score of our modal(one splites set)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
pop = model.score(x_test,y_test)
#print(df12)
def pridic_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns == location)[0][0]

    xx =np.zeros(len(x.columns))
    xx[0] = sqft
    xx[1] = bath
    xx[2] = bhk
    if loc_index >=0:
        xx[loc_index] = 1
    return model.predict([xx])[0]

pred = pridic_price("1st Phase JP Nager", 1000,2,2)
print(pred)
