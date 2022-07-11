a = input("any key to leave")
confirm = input("sure ? (Y/N)")
while confirm != "Y":
    confirm = input("sure ? (Y/N)")
for i in range(1,11):
    a = input("vraiment sur ({}/10)".format(i))