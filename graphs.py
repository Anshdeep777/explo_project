import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('ML_Ready_Lunar_Data.csv')
df.columns = df.columns.str.strip()

vacuum = df[df['Lunar Contact'] == 'Yes']
air    = df[df['Lunar Contact'] == 'No']

temp_cof_vac = vacuum.groupby('Temperature (°C)')['COF'].mean()
temp_cof_air = air.groupby('Temperature (°C)')['COF'].mean()  # ← fixed: was "team_cof_air"

plt.figure(figsize=(10, 5))
plt.plot(temp_cof_vac.index, temp_cof_vac.values, 'b-o', label='Lunar Vacuum')
plt.plot(temp_cof_air.index, temp_cof_air.values, 'r-o', label='Earth Air')  # ← fixed here too
plt.title('COF vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Average COF')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cof_vs_temp.png', dpi=150)
plt.show()