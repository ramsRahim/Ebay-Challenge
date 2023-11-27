def compare_tsv(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        line_num = 1
        differences = 0

        for line1, line2 in zip(file1, file2):
            if line1 != line2:
                print(f"Difference found at line {line_num}:")
                print(f"File 1: {line1.strip()}")
                print(f"File 2: {line2.strip()}")
                differences += 1

            line_num += 1

        # Check if files have different number of lines
        remaining_lines_file1 = list(file1)
        remaining_lines_file2 = list(file2)

        if remaining_lines_file1 or remaining_lines_file2:
            print(f"Files have different number of lines starting from line {line_num}.")
            differences += max(len(remaining_lines_file1), len(remaining_lines_file2))

        if differences == 0:
            print("Files are identical.")
        else:
            print(f"There are {differences} differences in total.")

# Example usage
file1 = 'submissions/output.tsv'
file2 = 'submissions/output1.tsv'

compare_tsv(file1, file2)
