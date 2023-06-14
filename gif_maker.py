import imageio
import os

filenames = os.listdir("figs")
filenames_sorted = sorted(filenames, key=lambda x: int(x.split("-")[1].split(".")[0]))
print(filenames_sorted)
filenamse_dir = []
for i in filenames_sorted:
    filenamse_dir.append(os.path.join("figs", i))

print(filenamse_dir)

with imageio.get_writer(os.path.join("figs", "dish.gif"), mode="I") as writer:
    for filename in filenamse_dir:
        image = imageio.imread(filename)
        writer.append_data(image)
