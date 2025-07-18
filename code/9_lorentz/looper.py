import os
import subprocess
import avfet_modules.terminal_options as terminal_options



'''
Parameters
'''
file = terminal_options.get("file", type=str, default="9_lorentz/im")



'''
Loop
'''
# Arrays
i_dt_arr = [(i, 2**(-i)) for i in range(4, 10)]

# Iterate over the timestep sizes and run the script with each value
for i_dt in i_dt_arr:
    folder = file + "/" + str(i_dt[0])
    os.makedirs("output/" + folder, exist_ok=True)
    subprocess.run([
        "python", "code/" + file + ".py",
        "--dt", str(i_dt[1]),
        "--folder", folder
    ])
