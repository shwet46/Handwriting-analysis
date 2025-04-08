import os
import extract  # Ensure extract.py with a function 'start(filename)' is in the same directory

def main():
    # Step 1: Change directory to 'images' and collect all file names
    os.chdir("images")
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    os.chdir("..")

    # Step 2: Check if 'raw_feature_list' file exists and read already processed page IDs
    page_ids = []
    raw_feature_path = "raw_feature_list"

    if os.path.isfile(raw_feature_path):
        print("Info: 'raw_feature_list' already exists. Loading processed file names...")
        with open(raw_feature_path, "r", encoding="utf-8") as label:
            for line in label:
                content = line.strip().split()
                if content:
                    page_id = content[-1]
                    page_ids.append(page_id)

    # Step 3: Process new files and append their extracted features
    with open(raw_feature_path, "a", encoding="utf-8") as label:
        count = len(page_ids)
        total_files = len(files)

        for file_name in files:
            if file_name in page_ids:
                continue

            try:
                features = extract.start(file_name)  # extract.start should return a list of features
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

            features.append(file_name)  # Add the file name at the end
            label.write('\t'.join(map(str, features)) + '\n')

            count += 1
            progress = (count * 100) / total_files
            print(f"{count} {file_name} {progress:.2f}%")

    print("Done!")

if __name__ == "__main__":
    main()