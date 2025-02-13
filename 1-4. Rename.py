import os

def rename_files_in_folder(folder_path, name_variable):
    # 폴더 존재 여부 확인
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # 폴더 내 모든 파일 가져오기
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 파일 정렬
    files.sort()

    for file in files:
        file_extension = os.path.splitext(file)[1]  # 파일 확장자 가져오기
        i = 1
        while True:
            new_name = f"{name_variable}_{i}{file_extension}"
            new_path = os.path.join(folder_path, new_name)
            if not os.path.exists(new_path):
                break
            i += 1

        old_path = os.path.join(folder_path, file)

        # 파일 이름 변경
        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} files in the folder: {folder_path}")

# 실행
rename_files_in_folder(
    folder_path="./Video/2. danger",
    name_variable="danger"
)
