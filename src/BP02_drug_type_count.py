import os
from tabulate import tabulate


def count_drug_names(dataset_path):
    """
    Count the number of each protein types (drugs).
    """
    drug_counts = {}
    file_count = [f.split('_')[1] for _, _, files in os.walk(dataset_path) for f in files]
   
    for drug in set(file_count):
        drug_counts[drug] = file_count.count(drug)
    return drug_counts


if __name__ == '__main__':

    dataset_paths = [
        '../data/processed_images_V2/heat_stress',
        '../data/processed_images_V2/mech_stress']

    for path in dataset_paths:
        drug_counts = count_drug_names(path)

        if path == dataset_paths[0]:
            print("Heat:")
        else:
            print("\nMech:")

        print(f"Total num of drugs: {sum(drug_counts.values())}")
        table_data = [["Drug", "Count"]]
        for drug, count in drug_counts.items():
            table_data.append([drug, count])
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
