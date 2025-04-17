from IPython.display import display
import pandas as pd

df = pd.read_csv(r"D:\ekstrak_fitur_1\ekstrak_fitur_1\dataset\phishing_1.csv")
display(pd.DataFrame(df.columns, columns=["Nama Kolom"]))
