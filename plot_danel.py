import pandas as pd
import json
import matplotlib.pyplot as plt

# load the data from the CSV file
df = pd.read_csv("danl_resume_vectors.csv")

plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    sensor = row["sensor_ane"]

    freqs = json.loads(row["Frequency_MHz"])
    danl = json.loads(row["Mean_DANL_dBFS"])

    plt.plot(
        freqs,
        danl,
        label=sensor,
        marker="o",
        markersize=3,
        linewidth=1,
    )

plt.xlabel("Frequency (MHz)")
plt.ylabel("Mean DANL (dBFS)")
plt.title("DANL by ANE Sensor")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
