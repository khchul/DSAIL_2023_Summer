base_file = open(file = 'modified_u1.base', mode = 'w')
test1 = open(file = 'u1_1-50.test', mode = 'w')
test2 = open(file = 'u1_51-.test', mode = 'w')
user = [0] * 944

lines = []

with open("u1.base") as i_file:
    for line in i_file:
        id, *others = map(int, line.split())
        user[id] += 1
        if id > 50:
            base_file.write(line)
        elif user[id] <= 15:
            base_file.write(line)
        else:
            lines.append(line)
            
with open("u1.test") as i_file:
    for line in i_file:
        id, *others = map(int, line.split())
        if id <= 50:
            lines.append(line)
        else:
            test2.write(line)

base_file.close()
test2.close()

sorted_lines = sorted(lines, key=lambda line: (int(line.split()[0]), int(line.split()[1])))

for line in sorted_lines:
    test1.write(line)

test1.close()