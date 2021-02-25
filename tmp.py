

path="./data/cub_200_2011/hierarchy/test_images_4_level_V1.txt"

d = {}
with open(path, "r") as f:
    for line in f.readlines():
        cls, genus, family, order = line.strip().split(' ')[1:]
        d[int(cls)] = f"{cls} {genus} {family} {order}"


with open("data/cub_200_2011/level.txt", "w") as f:
    for i in range(1,201):
        f.write(d[i]+"\n")
