# 读取文件内容
with open('OGB_MAG2500/P_Fea.txt', 'r') as file:
    file_content = file.read()

# 将小写 "m" 替换为大写 "M"
file_content = file_content.replace("'", '"')

# 将替换后的内容写入文件
with open('OGB_MAG2500/P_Fea.txt', 'w') as file:
    file.write(file_content)
