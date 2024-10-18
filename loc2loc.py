# 读取文件内容
with open('PubMed_Mini/GD.txt', 'r') as file:
    file_content = file.readlines()

# 对每一行中的元素进行位置互换
for i in range(len(file_content)):
    elements = file_content[i].split()  # 按空格分割元素
    if len(elements) == 2:  # 确保每行有两个元素
        new_line = elements[1] + " " + elements[0] + "\n"  # 互换位置
        file_content[i] = new_line

# 将修改后的内容写回文件
with open('PubMed_Mini/DG.txt', 'w') as file:
    file.writelines(file_content)