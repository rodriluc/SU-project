
testfile = "../datasets/membrane-beta_4state.3line.txt"

listID = []
listaa = []
list_filename = []

with open(testfile) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::3]
listaa = lines[1::3]
list_filename = lines[0::3]

for i in range(len(listID)):
    with open(list_filename[i]+".fasta", "w") as fn:
        fn.write(listID[i])
        fn.write(("\n"))
        fn.write(listaa[i])
