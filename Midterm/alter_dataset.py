"""
The purpose of this script is to specifically add the "targets" column to the mutations.csv file so that we can load it with numpy.
"""

with open("../mutations.csv", "r") as f:
    lines = f.readlines()

lines[0] = "targets" + lines[0]

with open("../mutations.csv", "w") as f:
    f.writelines(lines)
